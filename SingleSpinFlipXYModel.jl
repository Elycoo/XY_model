module SingleSpinFlip

const TDIMS=0         # Temporal dimension
const NDIMS=1         # Spatial dimension
using Statistics
using Base.Cartesian

# Hop to nearest-neighbor site

function hop(index::CartesianIndex{NDIMS},dir::Int64,lr::Int64,dims::NTuple{NDIMS,Int64}) where {NDIMS}
    # update index
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
    # size
    L::Int64
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
end

# Initialization

function IsingData(sim_data::SimData,start::Bool)
    if start # cold start
        IsingData(
            sim_data,
            0.0 * ones(ntuple(k->sim_data.L,NDIMS )),
            NDIMS*sim_data.L^NDIMS,
            # initial energy (-1)*Nz/2 if triangular lattice z is more then 2*NDIMS
            MeasureData(sim_data)
        )
    else  # hot start
        IsingData(
            sim_data,
            2*pi*rand(ntuple(k->sim_data.L,NDIMS )...),
            0.0, # approxmiated initial energy
            MeasureData(sim_data)
        )
    end

end

#  Compute single flip $\Delta E$
on_two_site_E(θ1::Float64,θ2::Float64) = -cos(θ1 - θ2)
function calc_delta_E(ising_lat::Array{Float64,NDIMS},pos::CartesianIndex, dθ::Float64)
    E_i = E_f = 0.0
    # left right
    for lr in 1:2
        # dims
        for d in 1:NDIMS
            # hop to nn
            nn=hop(pos,d,lr,size(ising_lat))
            E_i += on_two_site_E(ising_lat[nn],ising_lat[pos])
            E_f += on_two_site_E(ising_lat[nn],ising_lat[pos]+ dθ)
        end
        # add diagonal dimension
        # hop to nn and then to nn in the diagonal

        # ds = 1:NDIMS
        # nn=hop(pos,ds[1],lr,size(ising_lat))
        # nn=hop(nn,ds[2],lr,size(ising_lat))
        # E_i += delta_E_func(ising_lat[nn],ising_lat[pos])
        # E_f += delta_E_func(ising_lat[nn],ising_lat[pos]+ dθ)
    end
    return E_f-E_i
end

#  Single step

function next_step!(ising_data::IsingData)
    J=ising_data.sim_data.J
    ising_lat=ising_data.ising_lat
    # current_energy = ising_data.total_energy
    # flip site
    flip_site=rand(CartesianIndices(ising_lat))
    # calculate ratio
    α = pi/2
    suggested_dθ = 2*α*(rand()-0.5)
    delta_E=calc_delta_E(ising_lat,flip_site, suggested_dθ)
    ratio=exp(-J*delta_E)
    # accept or reject
    if ratio>rand()
        # flip
        ising_lat[flip_site]=ising_lat[flip_site]+suggested_dθ
        ising_data.total_energy += -delta_E
    end
end

# Direct calculation of energy of the system for 1D

function lat_energy(ising_data::IsingData)
    ising_lat = ising_data.ising_lat
    energy = on_two_site_E(ising_lat[1], ising_lat[end])
    for i in 1:(length(ising_lat)-1)
        energy += on_two_site_E(ising_lat[i], ising_lat[i+1])
    end
    energy

end # lat_energy

# Make a measurement

function make_measurement!(ising_data::IsingData,i)
    lat_size=length(ising_data.ising_lat)
    # average magnetization
    # magnetization vector
    mag_vec = [sum(cos.(ising_data.ising_lat)), sum(sin.(ising_data.ising_lat))]
    ising_data.measure_data.mag[i,:]=mag_vec/lat_size
    # magnetization vector squre
    mag_squre = sum(cos.(ising_data.ising_lat))^2 + sum(sin.(ising_data.ising_lat))^2
    ising_data.measure_data.mag[i,1]=mag_squre/lat_size^2
    # energy density
    ising_data.measure_data.energy[i]=ising_data.total_energy/lat_size
end

# MCMC run

function run_mcmc(sim_data::SimData,start::Bool)
    ising_data=IsingData(sim_data, start)
    lat_size=length(ising_data.ising_lat)
    # thermalize
    for i in 1:sim_data.num_thermal
        # sweep
        for j in 1:lat_size
            next_step!(ising_data)
        end
    end
    # measure
    for i in 1:sim_data.num_measure
        # sweep
        for j in 1:lat_size
            next_step!(ising_data)
        end
        make_measurement!(ising_data,i)
    end
    ising_data.ising_lat = mod.(ising_data.ising_lat,2*pi)
    ising_data
end

end # SingleSpinFlip module

#%%
using SpecialFunctions: besseli

function exact_energy(beta, L)
    besseli(1,beta)/besseli(0,beta)
end
#%%
using Plots
using Statistics
using LaTeXStrings
using Random
# Random.seed!(12463)
gr()
# Ts = [0.2,0.6, 0.9, 1.0, 1.05, 1.1, 1.15,1.2,1.4,2,4] ;betas= 1 ./Ts
betas=range(0.1, length=5,stop=2)
# Ls=[5,10,20]
Ls=[10]
num_measure=2^17
num_thermal=10000
start_cold = true
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
        sim_data=SingleSpinFlip.SimData(b,L,num_measure,num_thermal)
        global res=SingleSpinFlip.run_mcmc(sim_data,start_cold)    # start with all spins at the same direction
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
    plot!(fig_en, betas, ens, yerr=ens_std, xlabel=L"\beta",ylabel=L"E",label="mc L=$L",legend=:bottomright)
    # just in 1D:
    # plot!(fig_en,betas,[exact_energy(b,L) for b in betas],label="exact L=$L",legend=:bottomright)
    # plot!(fig_en,betas,[exact_energy(b,L) for b in betas],label="exact L=$L",legend=:topleft)

    plot!(fig_heat_c,betas,L^SingleSpinFlip.NDIMS*heat_c,xlabel=L"\beta",ylabel=L"C_{V}",label="mc L=$L",legend=:topright)

    plot!(fig_mag,betas,mags,yerr =mags_std,xlabel=L"\beta",ylabel=L"m",label="L=$L",legend=:topleft)
    # plot!(fig_mag,1 ./ betas,mags,yerr=mags_std,xlabel=L"k_{B}T/J",ylabel=L"m",label="L=$L",legend=:topleft)

    plot!(fig_tau,betas,taus,label="L=$L",xlabel=L"\beta",ylabel=L"\tau",legend=:topleft)
end
fig=plot(fig_en,fig_heat_c,fig_mag,fig_tau,layout=(1,4),size = (1000, 1600))
fig=plot(fig_en,fig_heat_c,fig_mag,fig_tau,size = (800, 800))
# fig=plot(fig_mag,size = (400, 600))
display(fig)

#%% testing
# SingleSpinFlip.NDIMS == 1 || throw("ERROR no way to calculate energy in 2 dim or up")
# v = [4.2, -0.76, 6.21, 7.31, 9.57]
# v = [0., 0., 0., 0., 0.]
# ee = -cos(v[1]-v[end])
# for i in 1:(length(v)-1)
#     global ee += -cos(v[i]-v[i+1])
# end
# @show ee
#
# sim_data = SingleSpinFlip.SimData(0.01,5,1000,64000)
# ising_data = SingleSpinFlip.IsingData(sim_data, true)
# ising_data.ising_lat = v
# ee = SingleSpinFlip.lat_energy(ising_data)
#  @show ee
