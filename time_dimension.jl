module SingleSpinFlip

const TDIMS = 1         # Temporal dimension
const NDIMS = 0 + TDIMS   # Spatial + Temporal dimensions
using Statistics
using Base.Cartesian
using Distributions: Normal
using Plots

# Hop to nearest-neighbor site

function hop(index::CartesianIndex{NDIMS},dir::Int64,lr::Int64,dims::NTuple{NDIMS,Int64},) where {NDIMS}
    if lr == 1
        # lr=1 -> plus 1
        hop_index = index[dir] == dims[dir] ? 1 : index[dir] + 1
    else
        # lr=2 -> minus 1
        hop_index = index[dir] == 1 ? dims[dir] : index[dir] - 1
    end
    # generate a new CartesianIndex with updated index
    CartesianIndex(Base.setindex(Tuple(index), hop_index, dir))

end

# Binning + (optionally) bootstrap analysis
using Bootstrap
function bin_bootstrap_analysis(data;min_sample_size = 128,func_boot = nothing,n_boot = 1000)
    # get total length
    data_size = length(data)
    # chop to closest power of 2
    chopped_data_size = 2^floor(Int, log(2, data_size))
    chopped_data =
        collect(Iterators.take(Iterators.reverse(data), chopped_data_size))
    # full data std
    if func_boot == nothing
        stds = [std(chopped_data) / sqrt(chopped_data_size)]
    else
        # bootstrap
        bs = bootstrap(func_boot, chopped_data, BasicSampling(n_boot))
        stds = [stderror(bs)[1]]
    end
    bin_size = 2
    while min_sample_size < div(chopped_data_size, bin_size)
        # bin size
        length_bin = div(chopped_data_size, bin_size)
        # binned data
        binned = reshape(chopped_data, (bin_size, length_bin))
        mean_binned = mean(binned, dims = 1)'
        # bin std
        if func_boot == nothing
            std_bin = std(mean_binned) / sqrt(length_bin)
        else
            # bootstrap
            bs = bootstrap(func_boot, mean_binned, BasicSampling(n_boot))
            std_bin = stderror(bs)[1]
        end
        # double bin size
        bin_size = bin_size * 2
        push!(stds, std_bin)
    end
    stds
end

# ## MCMC simulation

# Simulation parameters
mutable struct SimData
    # beta
    β::Float64
    # spatial size
    L::Int64
    # time dimension length
    M::Int64
    # epsilon
    eps::Float64
    # multiplication constant to ballance between open/close states
    c::Float64
    c_inv::Float64
    # numbers of measurements
    num_measure::Int64
    # numbers of themalization steps
    num_thermal::Int64
end

# Simulation data structure
mutable struct IsingData
    # simulation data
    sim_data::SimData
    # ising configuration
    ising_lat::Array{Float64,NDIMS}
    # energies
    energies::Array{Float64,1}
    #### worm information ####
    # the worm angle
    θ_worm::Float64
    # worm index
    pos_worm::CartesianIndex{NDIMS}
    # whether or not the worm is closed
    is_closed::Bool

end

# Initialization
function IsingData(sim_data::SimData)
    IsingData(
        sim_data,
        zeros(ntuple(n -> n < NDIMS ? sim_data.L : sim_data.M, NDIMS)), # lattice
        zeros(sim_data.num_measure), # prepare empty array for the energies
        # worm
        0.0, # angle
        CartesianIndex(1), # position
        true, # start with close
    )
end
"""
    mod2pi_(x)
Modulo ``2\\pi`` to range ``(-\\pi,\\pi)``

Example code:
```julia
julia> SingleSpinFlip.mod2pi_(3π-0.1)
3.0415926535897935

julia> SingleSpinFlip.mod2pi_(2π)
0.0

```
"""
function mod2pi_(x)
    x = mod(x, 2 * pi)
    x > pi ? x - 2 * pi : x
end
"""
    mean_mod2pi(x,y)
Make sure that the mean is taking right with mod ``2π``

Example code for testing:
```julia
x = rand(10).*2pi.-pi;
y = rand(10).*2pi.-pi;
scatter(x,1:10,label="",color=:red);
scatter!(y,1:10,label="",color=:red);
scatter!([SingleSpinFlip.mean_mod2pi(x[i],y[i]) for i in 1:10],1:10,label="")
```
"""
function mean_mod2pi(x::Float64, y::Float64)
    if abs(x - y) < pi
        return (x + y) / 2
    else
        return mod2pi_((x + 2 * pi + y) / 2)
    end
end

#  Single step
function next_step!(ising_data::IsingData)
    draw_move = rand(1:4)
    if draw_move == 1
        change_angle!(ising_data)
    elseif draw_move == 2
        if ising_data.is_closed
            open_worm!(ising_data)
        end
    elseif draw_move == 3
        if !ising_data.is_closed
            close_worm!(ising_data)
        end
    elseif draw_move == 4
        # return true
        if !ising_data.is_closed
            shift_worm!(ising_data)
        end
    end
end
function change_angle!(ising_data::IsingData)
    ising_lat = ising_data.ising_lat
    eps_ = ising_data.sim_data.eps

    # flip site
    flip_site = rand(CartesianIndices(ising_lat))
    flip_site_up = hop(flip_site, NDIMS, 1, size(ising_lat))
    flip_site_down = hop(flip_site, NDIMS, 2, size(ising_lat))

    if !ising_data.is_closed && flip_site == ising_data.pos_worm
        # if worm is open and we draw that position
        if rand(1:2) == 1
            # choosing the lattice angle
            mean_ = ising_lat[flip_site_down]
            sig = sqrt(eps_ / 2)
            d = Normal(mean_, sig)
            θ = rand(d)
            ising_lat[flip_site] = mod2pi_(θ)
            return true
        else
            # choosing the worm angle
            mean_ = ising_lat[flip_site_up]
            sig = sqrt(eps_ / 2)
            d = Normal(mean_, sig)
            θ = rand(d)
            ising_data.θ_worm = mod2pi_(θ)
            return true
        end
    elseif !ising_data.is_closed && flip_site_down == ising_data.pos_worm
        # if worm is open and we just above the worm
        mean_ = mean_mod2pi(ising_lat[flip_site_up], ising_data.θ_worm)
        d = Normal(mean_, sqrt(eps_ / 4))
        θ = rand(d)
        ising_lat[flip_site] = mod2pi_(θ)
        return true
    end
    mean_ = mean_mod2pi(ising_lat[flip_site_up], ising_lat[flip_site_down])
    d = Normal(mean_, sqrt(eps_ / 4))
    θ = rand(d)
    # always accept!
    ising_lat[flip_site] = mod2pi_(θ)
    return true
end

function open_worm!(ising_data::IsingData)
    ising_lat = ising_data.ising_lat
    eps_ = ising_data.sim_data.eps
    c = ising_data.sim_data.c

    # suggest to open
    pos = rand(CartesianIndices(ising_lat))
    npos = hop(pos, NDIMS, 1, size(ising_lat))

    # accept
    Δθ = mod2pi_(ising_lat[npos] - ising_lat[pos])
    A = min(1, c * exp((Δθ^2) / eps_))

    if A > rand()
        d = Normal(ising_lat[npos], sqrt(eps_ / 2))
        θ_worm = rand(d)

        ising_data.is_closed = false
        ising_data.pos_worm = pos
        ising_data.θ_worm = mod2pi_(θ_worm)
        true
    else
        false
    end
end

function close_worm!(ising_data::IsingData)
    # suggest to close
    ising_lat = ising_data.ising_lat
    eps_ = ising_data.sim_data.eps
    c_inv = ising_data.sim_data.c_inv

    pos = ising_data.pos_worm
    npos = hop(pos, NDIMS, 1, size(ising_lat))
    # accept
    Δθ = mod2pi_(ising_lat[npos] - ising_lat[pos])
    A = min(1, c_inv * exp(-(Δθ^2) / eps_))
    if A > rand()
        ising_data.is_closed = true
    else
        false
    end
end

function shift_worm!(ising_data::IsingData)
    ising_lat = ising_data.ising_lat
    eps_ = ising_data.sim_data.eps
    pos = ising_data.pos_worm

    up_or_down = rand(1:2)
    npos = hop(pos, NDIMS, up_or_down, size(ising_lat))

    # always accept!
    if up_or_down == 1
        # up
        d = Normal(ising_lat[pos], sqrt(eps_ / 2))
        θ_new = rand(d)

        ising_data.pos_worm = npos
        ising_data.θ_worm = ising_lat[npos]
        ising_lat[npos] = mod2pi_(θ_new)
    else
        # down
        d = Normal(ising_data.θ_worm, sqrt(eps_ / 2))
        θ_new = rand(d)

        ising_data.pos_worm = npos
        ising_lat[pos] = ising_data.θ_worm
        ising_data.θ_worm = mod2pi_(θ_new)
    end
    true
end

# MCMC run
function run_mcmc(sim_data::SimData)
    ising_data = IsingData(sim_data)
    lat_size = length(ising_data.ising_lat)
    # thermalize
    for i = 1:sim_data.num_thermal
        # sweep
        for j = 1:lat_size
            next_step!(ising_data)
        end
    end
    # measure
    count = 0
    for i = 1:sim_data.num_measure
        # sweep
        for j = 1:lat_size
            next_step!(ising_data)
        end

        if ising_data.is_closed
            calculate_energy!(ising_data, i)
            count += 1
        else
            ising_data.energies[i] = Inf
        end
    end
    # println(count / sim_data.num_measure)
    # deleting all the irrelevant entries
    e = ising_data.energies
    ising_data.energies = e[e.<Inf]
    return ising_data
end

time_direction_energy(θ1::Float64, θ2::Float64, eps_::Float64) = mod2pi_(θ1 - θ2)^2 / eps_^2
function calculate_energy!(ising_data::IsingData, index::Int64)
    e = 0.0
    ising_lat = ising_data.ising_lat
    eps_ = ising_data.sim_data.eps
    M = ising_data.sim_data.M
    β = ising_data.sim_data.β
    for i = 1:M
        j = i != 1 ? i - 1 : M
        e += time_direction_energy(ising_lat[i], ising_lat[j], 1.0)
    end
    ising_data.energies[index] = M / (2β) - (e * M) / β^2
end

function visualize(ising_data::IsingData)
    # SimData
    lat = ising_data.ising_lat
    M = ising_data.sim_data.M
    L = ising_data.sim_data.L
    if NDIMS == 2
        grid = vcat([[ind[1] ind[2]] for ind in CartesianIndices(lat)[:]]...)
        plot(xticks = 1:L, yticks = 1:M, gridopacity = 0.7)
        quiver!(
            grid[:, 1],
            grid[:, 2],
            quiver = (cos.(lat[:]), sin.(lat[:])) ./ 2,
            arrow = arrow(:closed, :head),
        )
        ylabel!("Time Direction")
        xlabel!("Spatial Direction")
    end
    if NDIMS == 1 && TDIMS == 1
        if ising_data.is_closed
            plot(lat, 1:M, label = false)
            scatter!(lat, 1:M, label = false)
        else
            pos = ising_data.pos_worm
            theta = ising_data.θ_worm
            rng = [1:pos[1], pos[1]:M, (pos[1]+1):M]
            plot(lat[rng[1]], rng[1], label = false, line = :black)
            plot!([theta, lat[rng[3]]...], rng[2], label = false, line = :black)
            scatter!(lat, 1:M, label = false, marker = :blue, ma = 0.5)
            scatter!([ising_data.θ_worm],[pos[1]],label = false,marker = :red)
        end
        xlims!((-pi, pi))
        xlabel!("θ")
        ylabel!("τ")
    end
end # visualize function

end # SingleSpinFlip module

#%%
function elip_Z(beta::Float64; cutoff = 5::Int64)
    l_cutoff = ceil(Int64, abs(cutoff * 2 * sqrt(1 / beta)))
    l_rng_2 = 0.25 * (-l_cutoff:l_cutoff) .^ 2
    sum(exp.(.-beta .* l_rng_2))
end

function elip_energy(beta::Float64; cutoff = 5::Int64)
    l_cutoff = ceil(Int64, abs(cutoff * 2 * sqrt(1 / beta)))
    l_rng_2 = 0.25 * (-l_cutoff:l_cutoff) .^ 2
    sum(l_rng_2 .* exp.(.-beta .* l_rng_2)) / sum(exp.(.-beta .* l_rng_2))
end


#%%
using Plots
default(
    titlefontsize = 18,
    legendfontsize = 15,
    guidefontsize = 15,
    tickfontsize = 15,
)
using Statistics
using LaTeXStrings
using Random
gr()
# Random.seed!(12463)
function run_sim()
    betas = range(0.1, length = 10, stop = 2.0)
    betas = range(1.0, length = 10, stop = 5.0)
    factors = [20.0]
    num_measure = 2^18
    num_thermal = 100000
    fig_en = plot(title = "energy")
    M = 0
    L = 1
    ens = Float64[]
    ens_std = Float64[]
    res = nothing
    for factor in factors
        ens = Float64[]
        ens_std = Float64[]
        Juno.progress() do id
        for (i,b) in enumerate(betas)
            c = 1.0
            M = floor(Int64, b * factor)
            eps_ = b / M
            sim_data = SingleSpinFlip.SimData(b,L,M,eps_,c,1 / c,num_measure,num_thermal)
            res = SingleSpinFlip.run_mcmc(sim_data)

            @info "Simulating β's" progress=i/length(betas) _id=id
            # to see the last state at each temperature uncomment the next line
            # display(SingleSpinFlip.visualize(res))

            # ENERGY
            push!(ens, mean(res.energies))

            # Standard Deviations
            stds = SingleSpinFlip.bin_bootstrap_analysis(res.energies)
            push!(ens_std, stds[end])

        end
        end
        plot!(fig_en,betas,ens,yerr = ens_std,
              xlabel = L"\beta",ylabel = L"E",label = "mc",legend = :topright)
        plot!(fig_en,betas,elip_energy.(Float64.(betas)),
              label = "exact",legend = :topright)
    end
    fig_en, betas, ens, ens_std, M, res
end
fig, betas, ens, ens_std, M, res = run_sim()
fig
#%% replot the same figure, the only diffrence is that here I don't need to run it all again
fig_en = plot(title = "energy")
rng = 1:length(betas)
plot!(fig_en,betas,ens,yerr = ens_std,
      xlabel = L"\beta",ylabel = L"E",label = "mc",legend = :topright)
plot!(fig_en,betas,elip_energy.(Float64.(betas)),
      label = "exact",legend = :topright)
