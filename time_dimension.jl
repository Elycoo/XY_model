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
    # coupling (in units of T)
    J::Float64
    # spatial size
    L::Int64
    # time dimension length
    M::Int64
    # numbers of measurements
    num_measure::Int64
    # numbers of themalization steps
    num_thermal::Int64
end

# Measurements data

mutable struct MeasureData
    # energy measurement time series
    energy::Array{Float64,1}
    # magnetization
    mag::Array{Float64,2}
end
function MeasureData(sim_data::SimData)
    MeasureData(zeros(sim_data.num_measure),zeros(sim_data.num_measure,2))
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
    #  whether or not the worm is closed
    is_closed::Bool

end

# Initialization
function IsingData(sim_data::SimData)
    IsingData(
        sim_data,
        1.0 * ones(ntuple(k->sim_data.L,NDIMS )),
        0,
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

#  Single step
function next_step!(ising_data::IsingData)
    draw_move = rand(2:3)
    change_angle!(ising_data)
    if draw_move == 1
    elseif draw_move == 2
        if ising_data.is_closed
            open_worm!(ising_data)
        end
    elseif draw_move == 3
        if !ising_data.is_closed
            close_worm!(ising_data)
        end
    elseif draw_move == 4
        if !ising_data.is_closed
            shift_worm!(ising_data)
        end
    end
end
function change_angle!(ising_data::IsingData)
    J=ising_data.sim_data.J
    ising_lat=ising_data.ising_lat
    eps_ = J/ising_data.sim_data.M

    # flip site
    flip_site=rand(CartesianIndices(ising_lat))
    flip_site_nn = []
    push!(flip_site_nn, hop(flip_site,NDIMS,1,size(ising_lat)) )
    push!(flip_site_nn, hop(flip_site,NDIMS,2,size(ising_lat)) )

    d = Normal(0, sqrt(eps_/2J))
    mean_ = (ising_lat[flip_site_nn[1]] +  ising_lat[flip_site_nn[1]])/2
    θ =  mean_ + rand(d)

    # always accept! (?)
    # flip
    ising_lat[flip_site] = mod2pi_(θ)
    true
end
#%
function open_worm!(ising_data::IsingData)
    J = ising_data.sim_data.J
    ising_lat = ising_data.ising_lat
    eps_ = J/ising_data.sim_data.M
    # suggeste to open
    d = Normal(0, sqrt(eps_/2J))
    θ_worm = rand(d)
    pos = rand(CartesianIndices(ising_lat))
    npos = hop(pos,NDIMS,2,size(ising_lat))
    # does not depend on the suggested θ_worm
    A = min(1,exp((ising_lat[npos]-ising_lat[pos])^2/(eps_/J)))
    if A > rand()
        ising_data.is_closed = false
        ising_data.pos_worm = pos
        ising_data.θ_worm = θ_worm
    end
    true
end
function close_worm!(ising_data::IsingData)
    # suggeste to close
    J = ising_data.sim_data.J
    ising_lat = ising_data.ising_lat
    eps_ = J/ising_data.sim_data.M

    pos = ising_data.pos_worm
    npos = hop(pos,NDIMS,2,size(ising_lat))
    A = min(1, exp(-(ising_lat[npos]-ising_lat[pos])^2/(eps_/J)))
    if A > rand()
        ising_data.is_closed = true
    end
    true
end

function shift_worm!(ising_data::IsingData)
    # return true
    println("I'm here")
    J = ising_data.sim_data.J
    ising_lat = ising_data.ising_lat
    eps_ = J/ising_data.sim_data.M

    d = Normal(0, sqrt(eps_/2J))
    θ_worm = rand(d)
    pos = ising_data.pos_worm
    up_or_down = rand(1:2)
    npos = hop(pos,NDIMS,up_or_down,size(ising_lat))
    A = true
    if A
        if up_or_down == 2
            # up
            ising_data.pos_worm = npos
            ising_data.θ_worm = ising_lat[npos]
            ising_lat[pos] = θ_worm
        else
            # down
            ising_data.pos_worm = npos
            ising_data.θ_worm = θ_worm
        end
    end
    true
end

# Make a measurement
function make_measurement!(ising_data::IsingData,i)
    lat_size=length(ising_data.ising_lat)
    # average magnetization
    # magnetization vector squre
    mag_squre = sum(cos.(ising_data.ising_lat))^2 + sum(sin.(ising_data.ising_lat))^2
    ising_data.measure_data.mag[i,1]=mag_squre/lat_size^2
    # energy density
    ising_data.measure_data.energy[i]=ising_data.total_energy/lat_size
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
    eps_ = ising_data.sim_data.J / ising_data.sim_data.M
    for i in 1:ising_data.sim_data.M
        j = i != 1 ? i - 1 : ising_data.sim_data.M
        e += time_direction_energy(ising_lat[i],ising_lat[j], eps_)
    end
    ising_data.total_energy = e * eps_
end

using Plots

function visualize(ising_data::IsingData)
    # SimData
    lat = ising_data.ising_lat
    L = ising_data.sim_data.L
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
        plot(lat,1:L)
        scatter!(lat,1:L)
        xlims!((-pi,pi))
        xlabel!("θ")
        ylabel!("τ")
    end
end # visualize function

end # SingleSpinFlip module

#%%
using SpecialFunctions: besseli
include("elip.jl")

function exact_energy(beta, L)
    # z = get_elip(exp(-beta))
    # e = -∂_beta (log(z));

    # e_β = exp(-beta)
    # e_β*get_der_elip(e_β)/get_elip(e_β)

    # -get_der_log_elip_exp(-beta)
    -get_der_elip(beta)/get_elip(beta)
    # 1/sqrt(beta)
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
# Ls=[5,10,20]
# betas = [1]
Ls=[20]
num_measure=2^18
num_thermal=10000
fig_en=plot(title="energy")
fig_heat_c=plot(title="heat capacity")
fig_mag=plot(title="magnetization")
fig_tau=plot(title="correlation time")
for L in Ls
    ens=Float64[]
    ens_std=Float64[]
    heat_c=Float64[]
    heat_c_std=Float64[]
    mags=Float64[]
    mags_std=Float64[]
    taus=Float64[]
    for b in betas
        sim_data=SingleSpinFlip.SimData(b,L,L,num_measure,num_thermal)
        global res=SingleSpinFlip.run_mcmc(sim_data)    # start with all spins at the same direction
        # global res=SingleSpinFlip.run_mcmc(sim_data,!start_cold)  # start with all spins  in random  direction

        # ENERGY
        push!(ens,mean(res.measure_data.energy))
        stds=SingleSpinFlip.bin_bootstrap_analysis(res.measure_data.energy)
        push!(ens_std,stds[end])

        # HEAT CAPACITY
        push!(heat_c, (mean(res.measure_data.energy.^2) - mean(res.measure_data.energy)^2)*b^2)

        # MAGNETIZATION
        mag = mean(res.measure_data.mag, dims=1)
        # push!(mags, sqrt(mag[1]^2+mag[2]^2))

        mag = mean(res.measure_data.mag[:,1])
        push!(mags, mag)

        stds=SingleSpinFlip.bin_bootstrap_analysis(res.measure_data.mag)
        push!(mags_std,stds[end])

        # CORRELATION TIME
        tau=0.5*((stds[end]/stds[1])^2-1)
        push!(taus,tau)
    end
    plot!(fig_en, betas, ens, yerr=ens_std, xlabel=L"\beta",ylabel=L"E",label="mc L=$L",legend=:topright)
    # just in 1D:
    plot!(fig_en,betas,[exact_energy(b,L) for b in betas],label="exact L=$L",legend=:topright)
    # plot!(fig_en,betas,[exact_energy(b,L) for b in betas],label="exact L=$L",legend=:topleft)

    plot!(fig_heat_c,betas,L^SingleSpinFlip.NDIMS*heat_c,xlabel=L"\beta",ylabel=L"C_{V}",label="mc L=$L",legend=:topright)

    plot!(fig_mag,betas,mags,yerr =mags_std,xlabel=L"\beta",ylabel=L"m",label="L=$L",legend=:topleft)
    # plot!(fig_mag,1 ./ betas,mags,yerr=mags_std,xlabel=L"k_{B}T/J",ylabel=L"m",label="L=$L",legend=:topleft)

    plot!(fig_tau,betas,taus,label="L=$L",xlabel=L"\beta",ylabel=L"\tau",legend=:topleft)
end
fig=plot(fig_en,fig_heat_c,fig_mag,fig_tau,layout=(1,4),size = (1000, 1600))
fig=plot(fig_en,fig_heat_c,fig_mag,fig_tau,size = (800, 800))
fig=plot(fig_en)
# fig=plot(fig_mag,size = (400, 600))
display(fig)

#%%
function run_(num_measure,num_thermal)
    b=1/1000
    b=1000
    L=20
    sim_data=SingleSpinFlip.SimData(b,L,L,num_measure,num_thermal)
    res=SingleSpinFlip.run_mcmc(sim_data)    # start with all spins at the same direction
    display(SingleSpinFlip.visualize(res))
    W = 0
    for i in 1:L
        j = i != 1 ? i - 1 : L
        W = W + SingleSpinFlip.mod2pi_(res.ising_lat[i] - res.ising_lat[j])*L/b
    end
    W = round(W/2*pi, digits=2)
    @show W
    @show res.is_closed
end
run_(num_measure,num_thermal)
#%%
b=10
L=20
sim_data=SingleSpinFlip.SimData(b,L,L,num_measure,num_thermal)
r = SingleSpinFlip.IsingData(sim_data)
SingleSpinFlip.MeasureData
