module SingleSpinFlip

const TDIMS=1         # Temporal dimension
const NDIMS=0+TDIMS   # Spatial + Temporal dimensions
using Statistics
using Base.Cartesian
using Distributions: Normal

# Hop to nearest-neighbor site

function hop(index::CartesianIndex{NDIMS},dir::Int64,lr::Int64,dims::NTuple{NDIMS,Int64}) where {NDIMS}
    # update index
    # @show index, dir, lr, dims
    if (lr==1)
        hop_index= index[dir]==dims[dir] ? 1 : index[dir]+1
    else
        hop_index= index[dir]==1 ? dims[dir] : index[dir]-1
    end
    # generate a new CartesianIndex with updated index
    CartesianIndex(Base.setindex(Tuple(index), hop_index, dir))

end

# Binning + (optionally) bootstrap analysis

using Bootstrap
function bin_bootstrap_analysis(data;min_sample_size=128,func_boot=nothing,n_boot=1000)
    # get total length
    data_size=length(data)
    # chop to closest power of 2
    chopped_data_size=2^floor(Int,log(2,data_size))
    chopped_data=collect(Iterators.take(Iterators.reverse(data),chopped_data_size))
    # full data std
    if func_boot==nothing
        stds=[std(chopped_data)/sqrt(chopped_data_size)]
    else
        # bootstrap
        bs = bootstrap(func_boot,chopped_data, BasicSampling(n_boot))
        stds=[stderror(bs)[1] ]
    end
    bin_size=2
    while min_sample_size < div(chopped_data_size,bin_size)
        # bin size
        length_bin=div(chopped_data_size,bin_size)
        # binned data
        binned=reshape(chopped_data,(bin_size,length_bin))
        mean_binned= mean(binned,dims=1)'
        # bin std
        if func_boot==nothing
            std_bin=std(mean_binned)/sqrt(length_bin)
        else
            # bootstrap
            bs = bootstrap(func_boot,mean_binned, BasicSampling(n_boot))
            std_bin = stderror(bs)[1]
        end
        # double bin size
        bin_size=bin_size*2
        push!(stds,std_bin)
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
    # numbers of measurements
    num_measure::Int64
    # numbers of themalization steps
    num_thermal::Int64
end

# Measurements data
mutable struct MeasureData
    # energy measurement time series
    energy::Array{Float64,1}
end
function MeasureData(sim_data::SimData)
    MeasureData(zeros(sim_data.num_measure))
end

# Simulation data structure
mutable struct IsingData
    # simulation data
    sim_data::SimData
    # ising configuration
    ising_lat::Array{Float64,NDIMS}
    # total energy
    total_energy::Float64
    # measurements data
    measure_data::MeasureData
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
        zeros(ntuple(n -> n < NDIMS ? sim_data.L : sim_data.M ,NDIMS )),
        0.0,
        # initial energy (-1)*Nz/2 if triangular lattice z is more then 2*NDIMS
        MeasureData(sim_data),
        0.0,
        CartesianIndex(1),
        true
    )
end

function mod2pi_(x)
    x = mod(x,2*pi)
    x > pi ? x-2*pi : x
end

# x = rand(10).*2pi.-pi
# y = rand(10).*2pi.-pi
# scatter(x,1:10,label="",color=:red);
# scatter!(y,1:10,label="",color=:red);
# scatter!([SingleSpinFlip.mean_mod2pi(x[i],y[i]) for i in 1:10],1:10,label="")
# check function ↓ with ↑
function mean_mod2pi(x::Float64, y::Float64)
    if abs(x-y) < pi
        return (x+y)/2
    else
        return mod2pi_((x+2*pi+y)/2)
    end
end


# Single step
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
    β=ising_data.sim_data.β
    ising_lat=ising_data.ising_lat
    eps_ = ising_data.sim_data.eps

	# flip site
    flip_site=rand(CartesianIndices(ising_lat))
    flip_site_up = hop(flip_site,NDIMS,1,size(ising_lat))
    flip_site_down = hop(flip_site,NDIMS,2,size(ising_lat))

    if !ising_data.is_closed && flip_site == ising_data.pos_worm
        # if worm is open and we draw that position
        if rand(1:2) == 1
            # choosing the lattice angle
            mean_ = ising_lat[flip_site_down]
            sig = sqrt(eps_/2)
            d = Normal(mean_, sig)
            θ = rand(d)
            ising_lat[flip_site] = mod2pi_(θ)
            return true
        else
            # choosing the worm angle
            mean_ = ising_lat[flip_site_up]
            sig = sqrt(eps_/2)
            d = Normal(mean_, sig)
            θ = rand(d)
            ising_data.θ_worm = mod2pi_(θ)
            return true
        end
    elseif !ising_data.is_closed && flip_site_down == ising_data.pos_worm
        # if worm is open and we just above the worm
        mean_ = mean_mod2pi(ising_lat[flip_site_up],
                            ising_data.θ_worm)
        d = Normal(mean_, sqrt(eps_/4))
        θ = rand(d)
        ising_lat[flip_site] = mod2pi_(θ)
        return true
    end
    mean_ = mean_mod2pi(ising_lat[flip_site_up],
                        ising_lat[flip_site_down])
                         # TODO - periodic boundary conditions! done!
    d = Normal(mean_, sqrt(eps_/4))
    θ = rand(d)
    # always accept!
    # flip
    ising_lat[flip_site] = mod2pi_(θ)
    return true

end
#%
function open_worm!(ising_data::IsingData)
    β=ising_data.sim_data.β
    ising_lat=ising_data.ising_lat
    eps_ = ising_data.sim_data.eps

    # suggest to open
    pos = rand(CartesianIndices(ising_lat))
    npos = hop(pos,NDIMS,1,size(ising_lat))

    # accept
    # can add C multiplication
	Δθ = mod2pi_(ising_lat[npos]-ising_lat[pos])
    A = min(1,exp((Δθ^2)/(eps_)))
    # TODO: mod2pi_ for the angle diffrence

    if A > rand()
        d = Normal(ising_lat[npos], sqrt(eps_/2))
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
    β=ising_data.sim_data.β
    ising_lat=ising_data.ising_lat
    eps_ = ising_data.sim_data.eps

    # suggeste to close
    pos = ising_data.pos_worm
    npos = hop(pos,NDIMS,1,size(ising_lat))
	Δθ = mod2pi_(ising_lat[npos]-ising_lat[pos])
    A = min(1, exp(-(Δθ^2)/(eps_)))
    if A > rand()
        ising_data.is_closed = true
    else
		false
	end
end

function shift_worm!(ising_data::IsingData)
    β=ising_data.sim_data.β
    ising_lat=ising_data.ising_lat
    eps_ = ising_data.sim_data.eps

	pos = ising_data.pos_worm
    up_or_down = rand(1:2)
    npos = hop(pos,NDIMS,up_or_down,size(ising_lat))

    # always accept!
    if up_or_down == 1
        # up
        d = Normal(ising_lat[pos], sqrt(eps_/2))
        θ_new = rand(d)

        ising_data.pos_worm = npos
        ising_data.θ_worm = ising_lat[npos]
        ising_lat[npos] = mod2pi_(θ_new)
    else
        # down
        d = Normal(ising_data.θ_worm, sqrt(eps_/2))
        θ_new = rand(d)

        ising_data.pos_worm = npos
        ising_lat[pos] = ising_data.θ_worm
        ising_data.θ_worm = mod2pi_(θ_new)
    end
    true
end

# Make a measurement
function make_measurement!(ising_data::IsingData,i)
    lat_size=length(ising_data.ising_lat)
    # energy density
    ising_data.measure_data.energy[i]=ising_data.total_energy
end

# MCMC run

function run_mcmc(sim_data::SimData)
    ising_data=IsingData(sim_data)
    lat_size=length(ising_data.ising_lat)
    # thermalize
    for i in 1:sim_data.num_thermal
        # sweep
        for j in 1:lat_size
            next_step!(ising_data)
        end
    end
    # measure
    count = 0
    for i in 1:sim_data.num_measure
        # sweep
        for j in 1:lat_size
            next_step!(ising_data)
        end

        if ising_data.is_closed
            calculate_energy!(ising_data)
            make_measurement!(ising_data, i)
            count += 1
        end
    end
    println(count/sim_data.num_measure)
    # ising_data.ising_lat = mod2p_.(ising_data.ising_lat)
    ising_data
end

time_direction_energy(θ1::Float64,θ2::Float64,eps_::Float64) = mod2pi_(θ1 - θ2)^2/eps_^2
function calculate_energy!(ising_data::IsingData)
    e = 0.0
    ising_lat = ising_data.ising_lat
    eps_ = ising_data.sim_data.eps
	M = ising_data.sim_data.M
	β = ising_data.sim_data.β
    for i in 1:M
        j = i != 1 ? i - 1 : M
        e += time_direction_energy(ising_lat[i],ising_lat[j], 1.0)
    end
    ising_data.total_energy = M/(2*β) - e * M / β^2
end

using Plots

function visualize(ising_data::IsingData)
    # SimData
    lat = ising_data.ising_lat
    L = ising_data.sim_data.L
	M = ising_data.sim_data.M
    if NDIMS  == 2
        grid = vcat([ [ind[1] ind[2]] for ind in CartesianIndices(lat)[:]]...)
        plot(xticks=1:L, yticks=1:L, gridopacity=0.7)
        quiver!(grid[:, 1],grid[:, 2],
                quiver=(cos.(lat[:]), sin.(lat[:])) ./ 2,
                arrow= arrow(:closed,:head))
        ylabel!("Time Direction")
        xlabel!("Spatial Direction")
    end
    if NDIMS == 1 && TDIMS == 1
        if ising_data.is_closed
            plot(lat,1:M,label=false)
            scatter!(lat,1:M,label=false)
        else
            pos = ising_data.pos_worm
            theta = ising_data.θ_worm
            rng = [1:pos[1], pos[1]:M, (pos[1]+1):M]
            plot(lat[rng[1]], rng[1],label=false,line=:black)
            plot!([theta, lat[rng[3]]...], rng[2],label=false,line=:black)
            scatter!(lat,1:M,label=false,marker=:blue,ma=0.5)
            scatter!([ising_data.θ_worm] ,[pos[1]],label=false,marker=:red)
        end
        xlims!((-pi,pi))
        xlabel!("θ")
        ylabel!("τ")
    end
end # visualize function

end # SingleSpinFlip module

#%%
function elip_energy(beta::Float64; cutoff=10::Int64)
    l_cutoff = ceil(Int64, abs(cutoff * sqrt(1/beta)))
    l_rng_2 = 0.25*(-l_cutoff:l_cutoff).^2
    sum(l_rng_2.*exp.(.-beta.*l_rng_2)) / sum(exp.(.-beta.*l_rng_2))
end

#%%
# TODO: Need to implemet different coupling in the temporal and the Spatial Directions.
using Plots
default(titlefontsize = 18,
        legendfontsize = 15,
        guidefontsize = 15,
        tickfontsize = 15)
using Statistics
using LaTeXStrings
using Random
# Random.seed!(12463)
gr()
betas=range(0.1, length=10,stop=2)
betas=range(1, length=10,stop=5)
Ls=[20]
num_measure=2^18
num_thermal=10000
ens=Float64[]
ens_std=Float64[]
for M in Ls
    for b in betas
		eps_ = b/M
        sim_data=SingleSpinFlip.SimData(b,1,M,eps_,num_measure,num_thermal)
        global res=SingleSpinFlip.run_mcmc(sim_data)    # start with all spins at the same direction
		# display(SingleSpinFlip.visualize(res))

        # ENERGY
        push!(ens,mean(res.measure_data.energy) )
        stds=SingleSpinFlip.bin_bootstrap_analysis(res.measure_data.energy)
        push!(ens_std,stds[end])
    end
end
fig_en=plot(title="energy")
plot!(fig_en, betas, ens, yerr=ens_std, xlabel=L"\beta",ylabel=L"E",label="mc",legend=:topright)
# just in 1D:
plot!(fig_en,betas,elip_energy.(betas),label="exact",legend=:topright)

#%%
fig_en=plot(title="energy")
plot!(fig_en, betas, ens, yerr=ens_std, xlabel=L"\beta",ylabel=L"E",label="mc",legend=:topright)
# just in 1D:
plot!(fig_en,betas,0.2elip_energy.(betas),label="exact",legend=:topright)
# plot!(fig_en,betas,[exact_energy(b,M) for b in betas],label="exact L=$M",legend=:topleft)
fig=plot(fig_en)
display(fig)

#%%
function run_(num_measure,num_thermal)
    b=1/1000
    b=1000
    M=20
    sim_data=SingleSpinFlip.SimData(b,1,M,num_measure,num_thermal)
    res=SingleSpinFlip.run_mcmc(sim_data)    # start with all spins at the same direction
    display(SingleSpinFlip.visualize(res))
    W = 0
    for i in 1:M
        j = i != 1 ? i - 1 : M
        W = W + SingleSpinFlip.mod2pi_(res.ising_lat[i] - res.ising_lat[j])*M/b
    end
    W = round(W/2*pi, digits=2)
    @show W
    @show res.is_closed
end
# run_(num_measure,num_thermal)
#%%
# b=10
# M=20
# sim_data=SingleSpinFlip.SimData(b,1,M,num_measure,num_thermal)
# r = SingleSpinFlip.IsingData(sim_data)
# SingleSpinFlip.MeasureData
