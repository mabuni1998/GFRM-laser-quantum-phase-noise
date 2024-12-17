
struct StochasticRateEquation
    A::Matrix{Int}
    parameters::Vector{Float64}
    rate_func::Function
    population_func::Function
end

StochasticRateEquation(A::Matrix{Int},parameters::Vector{Float64},rate_update::Function) = StochasticRateEquation(A,parameters,rate_update,update_population!)


function gfrm(eq::StochasticRateEquation,nsteps::Int;N0=nothing)
    N = zeros(Float64,size(eq.A,1))
    avgs = zeros(Float64,2*size(eq.A,1)+1)
    a = zeros(Float64,size(eq.A,2))

    gfrm_loop!(avgs,N,a,eq.A,nsteps,eq.parameters,eq.rate_func,eq.population_func,nothing,nothing,nothing,nothing,nothing)

    return Dict("averages" => avgs[1:end-1]/avgs[end],"t_end"=>avgs[end]) 
end

function gfrm_intra(eq::StochasticRateEquation,nsteps::Int;N0=nothing)
    N = zeros(Float64,size(eq.A,1))
    avgs = zeros(Float64,2*size(eq.A,1)+1)
    a = zeros(Float64,size(eq.A,2))

    if !isnothing(N0)
        N.=N0
    end

    data = zeros(Float64,(length(N),nsteps))
    timevec = zeros(Float64,nsteps)

    gfrm_loop!(avgs,N,a,eq.A,nsteps,eq.parameters,eq.rate_func,eq.population_func,data,timevec,nothing,nothing,nothing)

    return Dict("averages" => avgs[1:end-1]/avgs[end],"t_end"=>avgs[end],"time_series"=>data,"times"=>timevec) 
end


function gfrm_out(eq::StochasticRateEquation,nsteps::Int,fout::Function;N0=nothing)
    N = zeros(Float64,size(eq.A,1))
    avgs = zeros(Float64,2*size(eq.A,1)+1)
    a = zeros(Float64,size(eq.A,2))

    if !isnothing(N0)
        N.=N0
    end

    output_type = Float64
    for k in 1:size(eq.A,2)
        out,out_result = fout(N,k)
        if out_result
            output_type = typeof(out)
        end
    end

    data_out = zeros(output_type,nsteps)
    timevec_out = zeros(Float64,nsteps)
    outcouple_duration = zeros(Float64,nsteps+2)
    outcouple_nemitter = zeros(BigInt,nsteps+2)
    
    function outcouple!(data_out,timevec_out,outcouple_duration,N,avgs,k::Int,dt,idx::Int)
        output,event_result = fout(N,k)
        outcouple_duration[end-1] += dt*max(1,N[1])
        
        if event_result
            data_out[idx] = output
            timevec_out[idx] = avgs[end]
            
            outcouple_duration[idx] = outcouple_duration[end-1]
            outcouple_duration[end-1] = 0
            idx += 1

        end
        if k==5
            outcouple_duration[end-1] = 0
        end
        return idx
    end        
    gfrm_loop_outcouple!(avgs,N,a,eq.A,nsteps,eq.parameters,eq.rate_func,eq.population_func,nothing,nothing,data_out,timevec_out,outcouple_duration;fout=outcouple!)
    
    
    return Dict("averages" => avgs[1:end-1]/avgs[end],"t_end"=>avgs[end],"out_series"=>data_out,"out_times"=>timevec_out,"out_decay"=>outcouple_duration[1:end-2],"emitter"=>outcouple_nemitter[1:end-2]) 
end


function save_series!(data::Matrix,timevec::Vector{Float64},N::Vector{Float64},i::Int,t_tot::Float64)
    for j in 1:size(data,1)
        data[j,i] = N[j]
    end
    timevec[i] = t_tot 
end

function save_series!(d::Nothing,a::Nothing,N::Vector{Float64},i::Int,t_tot::Float64)
    return
end

function gfrm_loop!(avgs::Vector{Float64},N::Vector{Float64},a::Vector{Float64},A::Matrix{Int},nsteps::Int,params::Vector{Float64},
    update_rates!::Function,population_func!::Function,data,timevec,data_out,timevec_out,outcouple_duration;fout=(x,y,z,a,b,c,d)->nothing)
    event_times = zeros(Float64,size(A,2))
    rand_numbers = zeros(Float64,size(A,2))
    dt = 0
    k = 1
    prog = Progress(100, desc= "GFRM Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    i_mod = nsteps ÷ 100 

    for i in 1:nsteps
        if i % i_mod == 0
            next!(prog)
        end            
        update_rates!(a,N,params)
        rand!(rand_numbers)
        calc_eventtimes!(event_times,rand_numbers,a)
        dt,k = findmin(event_times)
        time_averages!(avgs,N,dt)
        fout(data_out,timevec_out,outcouple_duration,N,avgs,k,dt)
        population_func!(N,A,k,dt,params)
        save_series!(data,timevec,N,i,avgs[end])
    end
    return
end

function gfrm_loop_outcouple!(avgs::Vector{Float64},N::Vector{Float64},a::Vector{Float64},A::Matrix{Int},nsteps::Int,params::Vector{Float64},
    update_rates!::Function,population_func!::Function,data,timevec,data_out,timevec_out,outcouple_duration;fout=(x,y,z,a,b,c,d,e)->nothing)
    event_times = zeros(Float64,size(A,2))
    rand_numbers = zeros(Float64,size(A,2))
    dt = 0
    k = 1
    prog = Progress(100, desc= "GFRM Conv. Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    i_mod = nsteps ÷ 100 
    idx = 1
    prev = 1
    dt_prev = 0
    while idx <= nsteps
        if  idx != prev && idx % i_mod == 0
            next!(prog)
        end            
        update_rates!(a,N,params)
        rand!(rand_numbers)
        calc_eventtimes!(event_times,rand_numbers,a)
        dt_prev = dt
        dt,k = findmin(event_times)
        time_averages!(avgs,N,dt)
        prev = idx
        idx = fout(data_out,timevec_out,outcouple_duration,N,avgs,k,dt,idx)
        population_func!(N,A,k,dt,params) 
    end
    return
end

function update_population!(N,A,k)
    for i in eachindex(N)
        N[i] += A[i,k]
    end
    return
end

function time_averages!(avgs,N,dt)
    offset = length(N)
    for i in eachindex(N)
        avgs[i] += N[i]*dt
        avgs[i+offset] += N[i]^2*dt      
    end
    avgs[end] += dt
    return
end

function calc_eventtimes!(event_times,r,a)
    for i in eachindex(event_times)
        event_times[i] = - log(r[i])/a[i]
    end
    return
end
