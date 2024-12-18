# GFRM laser quantum phase noise
An implementation of Gillespies first reaction method for calculating quantum phase noise in laser rate equations used in REFERENCE PAPER.
See also [https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.253801](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.253801) for even more context.

# Installation

Download repository and add the Stochastic module to the current environemnt with the command (press ] to get to package manager):
```julia
pkg> dev Stochastic//
```
\
# Examples
In the following, we show showcase different functionalities of the code, which are necessary to reproduce the results in REFERENCE. 

## Setup problem
We first define the parameters. Here we consider only one particular pump rate, but by changing the index `i`, other pump rates can be investigated.
```julia
using Stochastic
using Statistics

#Define parameters (same as fig. 2 in main text of paper)
g,kappa,gamma_A,gamma_D,n0,alpha = 0.1,0.04,0.0012,1,5,0

#Define the pump rates and choose one by index i
pump_rates = 10 .^(range(-4,stop=log(20)/log(10),length=50))

i = 1
P = pump_rates[i]

#Define the decay rate and collect the parameters
gamma_r = 4*g^2 ./ (kappa+gamma_A+gamma_D + P)
params = [P,gamma_r,kappa,gamma_A,n0,alpha]
```

In the code, one defines the different events and their effect on the populations by giving a matrix `A`. The matrix should have dimensions (`rows` $\times$ `columns`) , where rows is the number of stochastic variables (here three: na, ne, and phi) and columns is the number of events (here 6).
```julia
#Define the matrix A that defines the popultion changes for each event.
A =[-1 -1  0  1  1  0; #Photon population change
     0  1 -1 -1 -1  1; #Emitter population change
     0  0  0  0  0  0]; #Phase population is unchanged
```

Then a function the gives the the rates at each timestep should be defined. Here, `a` is a vector of length `columns` that contains the rates, `N` is a vector of length `rows` containing the stochastic variables, and params are any parameters necessary for the simulation.
```julia
#Define how the 6 rates are updated (see eq. 1 and 2 in main text of paper)
function update_rates!(a,N,params)
    P,gamma_r,kappa,gamma_A,n0,alpha = params

    a[1] = kappa*N[1]                   #Cavity loss
    a[2] = gamma_r*N[1]*(n0-N[2])  #Stimulated absorption
    a[3] = gamma_A*N[2]                   #Non-radiative decay
    a[4] = gamma_r*N[2]*N[1]              #Stimulated emission
    a[5] = gamma_r*N[2]                   #Spontaneous emission
    a[6] = P*(n0 - N[2])     #Pump-event
    return
end
```

We then provide a function that updates the population (this function is run at each timestep). `k`, here denotes which event (index of column) that happened and the population is changed according to that. Note that `N[end]`, which is the phase phi, is changed only if `k==5`. All other variables (na and ne) are changed according to `A` (notice the loop over the rows of `A` at the column index `k`)
```julia
#Define how the population is updated. Here we also add the phase noise if k==5 (spontanoues decay)
function update_population!(N,A,k,dt,params)
    if k==5
        if N[1] == 0
            N[end] += 2*pi*(rand()-0.5)
        else
            phase_adjustment = angle(sqrt(N[1]) * exp(im * N[end]) + exp(2 * pi * (rand()-0.5) * im))
            N[end] += phase_adjustment - angle(exp(im * N[end]))
        end
    end
    # param[6] is alpha and the following line adds the phase noise due to refractive index changes
    N[end] += params[6]*N[2]*params[2]*dt
    
    #Update the number of photons and emitters
    for i in 1:length(N)-1
        N[i] += A[i,k]
    end
    return
end
```

Combining all of this, we can define a stochastic rate equation problem:

```julia
#Define stochastic rate equation problem
prob = StochasticRateEquation(A,params,update_rates!,update_population!)    
```

## Getting outcoupled field and calculate spectrum
When solving the problem, we can provide an output function that specifies which events should be recorded as outcoupled events (here `k=1` as it corresponds to a cavity decay event):
```julia
#Define the output function. Here we output the phase of the cavity field if k==1 (cavity decay event)
function fout(N,k)
    if k==1
        N[end],true
    else
        nothing,false
    end
end
```

We then solve the problem using Gillespies First Reaction Method (GFRM) and specify that we want 1 million outcoupling events defined by `fout`:
```julia
#Solve the problem and output the result. Here we require 1_000_000 outcoupling events.
result_out = gfrm_out(prob,1_000_000,fout)
```

From the solution we can extract first and second order statistical moments:
```julia
#First order moments of the stochastic variables (na,ne and phi) are stored in result_out["averages"][1:3]
pops = result_out["averages"]
na = pops[1]
ne = pops[2]
phi_avg = pops[3]

#Second order moments (<na^2>,<ne^2> and <phi^2>) are stored in pops[4:6]
#We calculate the second order correlation function g2
g2 = (pops[4] - pops[1])/pops[1]^2
```

We can also see the outcoupled electric field as a function of time:

```julia
#The outcoupled series (phase at each outcoupling event) is stored in result_out["out_series"] and the corresponding times in result_out["out_times"]
using PyPlot
pygui(true)
fig,ax = subplots(1,1,figsize=(4.5,4.5))
ax.plot(result_out["out_times"][1:100],real.(exp.(im .* result_out["out_series"][1:100])))
ax.set_xlabel("Time")
ax.set_ylabel("Outcoupled Field")
plt.tight_layout()
savefig("outcoupled_field.jpg")
```
![outcoupledfield](outcoupled_field.jpg?raw=true)

The duration of each event is stored in `result_out["out_decay"]` and we can calculate the mean outcoupled duration as:

```julia
julia> mean(result_out["out_decay"])
4.407
```

For the actual spectrum and linewidth we specify the appropiate frequency range and calculate the emission spectrum assuming each pulse to have length of `mean(result_out["out_decay"])`. We could have also input just  `result_out["out_decay"]` to get varying pulse durations.:
```julia 
#We choose an appropiate frequency range. We here choose 10 times the schawlow-townes linewidth
st_lw = (pops[2]) .* gamma_r ./ (2*pops[1])/(2*pi)
freq = range(-10*st_lw,10*st_lw,length=201)

spec_out = calc_spec(result_out["out_times"],exp.(im .* result_out["out_series"]),mean(result_out["out_decay"]),freq;mode=:lor)
#Equivalently one can assume variable durations of outcoupling events and use a sinc function to calculate the spectrum
#spec_out = Stochastic.exponential_decay_fourier(result_out["out_times"],result_out["out_series"],result_out["out_decay"],freq;mode=:sinc)
```

This gives the spectrum:

```julia
#Extract the linewidth with a cumulative lorenzian fit
fit_x_c,_, B,fit_y_c = fit_lorr_cum(freq,spec_out,st_lw)

#Plot the spectrum
fig,ax = subplots(1,1,figsize=(4.5,4.5))
ax.plot(freq,spec_out ./ max(spec_out...),"r-",label="stoch fit")
#Plot fit 
ax.plot(fit_x_c,fit_y_c ./ max(fit_y_c...),"bo",label="Fit")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectrum")
ax.legend()
savefig("spec.jpg")
```

![spec](spec.jpg?raw=true)


## Population sweep
If one wants only the averages. One can use just the gfrm solver. This is more efficient for sweeping over parameters since a timeseries is not saved.
Here we sweep over the pump rates and calculate the average number of photons in the cavity
```julia

na_list = zeros(length(pump_rates))
for (i,P) in enumerate(pump_rates)
    gamma_r_avg = 4*g^2 ./ (kappa+gamma_A+gamma_D + P)
    params_avg = [P,gamma_r_avg,kappa,gamma_A,n0,alpha]
    prob_avg = StochasticRateEquation(A,params_avg,update_rates!,update_population!)    
    result_avg = gfrm(prob_avg,1_000_000)
    na_list[i] = result_avg["averages"][1]
end
#Plot the number of photons
fig,ax = subplots(1,1,figsize=(4.5,4.5))
ax.loglog(pump_rates,na_list)
ax.set_xlabel("Pump rate")
ax.set_ylabel("Number of photons")
plt.tight_layout()
savefig("na_sweep.jpg")
```

![nas](na_sweep.jpg?raw=true)

## Calculate RIN
Here, we provide the code necessary to calculate the intra and outer cavity RIN, which follows https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.253801 closely.


If one wants the time series of the intra cavity field one can use the `gfrm_intra` solver

```julia
result_in = gfrm_intra(prob,1_000_000)
```

The interpolated intra cavity field and RIN is then:
```julia
#Interpolate the intra cavity field to a fixed time step
dt = 10.0
photons_in = result_in["time_series"][1,:]
photons_in_interpolated = interpolate_previous(result_in["times"],photons_in,dt)
rin_in,freq_rin_in = RIN(photons_in_interpolated,dt*1e-12,result_in["averages"][1];N = 10000)
```

Similarly, the outer cavity RIN is given as:

```julia
#The outcoupled RIN is similarly given (we have the outcoupled field from above and just need to interpolate it)
photons_out_interpolated = interpolate_outcoupled(result_out["out_times"],abs.(exp.(im .* result_out["out_series"])),dt)
rin_out,freq_rin_out = RIN(photons_out_interpolated,dt*1e-12,sum(photons_out_interpolated)/length(photons_out_interpolated);N = 10000)
```

Finally, we can plot them both together with the shot noise limit:

```julia
#Plot the RIN
fig,ax = subplots(1,1,figsize=(4.5,4.5))
ax.plot(freq_rin_in[2:end÷2] * 10^-9,10/log(10)*log.(rin_in[2:end÷2]),"b",label="Intra cavity RIN")
ax.plot(freq_rin_out[2:end÷2]* 10^-9,10/log(10)*log.(rin_out[2:end÷2]),"r",label="Outcoupled RIN")
ax.axhline(10/log(10)*log.(2/(kappa*result_out["averages"][1])*1e-12),color="black",ls="--",label="Shot noise limit")
ax.legend()
xlabel("Frequency [GHz]")
ylabel("RIN [dB/Hz]")
plt.tight_layout()
savefig("rin.jpg")
```

![rin](rin.jpg?raw=true)
