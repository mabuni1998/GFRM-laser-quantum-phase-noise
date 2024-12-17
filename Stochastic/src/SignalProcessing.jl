using FFTW
using Statistics
using Interpolations
using LsqFit
using LinearAlgebra

function RIN(y,dt,np_av;N = 1)
    nelements = length(y)
    M = nelements÷N
    if rem(M,2) != 0
        M = M-1
    end
    K = floor(Int,nelements/M)
    Wt_segments = (M-1)*dt

    spec_temp = zeros(ComplexF64,div(M,2)+1)
    psd = zeros(Float64,M)
    f = plan_rfft(psd)

    fft_bartlet!(psd,spec_temp,y,f,K,M,Wt_segments,dt)
    freq = rfftfreq(M,1/dt)
    RF = 2*psd[1:div(M,2)+1]/np_av^2

    return RF,freq
end

function fft_bartlet!(psd::Vector{Float64},spec_temp::Vector{ComplexF64},y,f,K::Int,M::Int,Wt_segments::Float64,dt::Float64)
    prog = Progress(100,  desc= "FFT Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    i_mod = max(K÷100,1)
    for i in 1:K
        if i%i_mod == 0
            next!(prog)
        end
        mul!(spec_temp,f,view(y,1+(i-1)*M:i*M))
        for j in eachindex(psd) 
            @inbounds psd[j] += abs(spec_temp[j]*dt)^2/Wt_segments/K
        end
    end
end

function fit_lorr(x,y,fmin=-Inf,fmax=Inf,B0=nothing)
    model(x,p) = p[1] ./ ((x .-p[3]) .^ 2 .+ (p[2]/2)^2) .+ p[4]

    mask = (fmin .<= x .<= fmax) 
    x_in = view(x,mask)
    y_in = view(y,mask)

    A0 = max(y_in...)
    if isnothing(B0)
        mask_start = findlast(x->x<A0/2,y)
    
        fwhm = 2 * abs(x[mask_start])
    
        B0=fwhm
    else 
        B0 = B0
    end
    #println()
    fit_result = curve_fit(model, x_in, y_in, [A0,B0,x_in[y_in .== max(y_in...)][1],min(y_in...)])
    A0,B0,C0,D0 = coef(fit_result)
    println("Fit error: $(mse(fit_result))")
    println([A0,B0,C0,D0])
    return x_in,model(x_in,[A0,B0,C0,D0]),B0
end


function fit_lorr_cum(x,y,B)
    dw = x[2]-x[1]
    L_error = dw *cumsum(y)

    s = B
    d = 0
    A = max(y...)/2
    c = L_error[end]-A*s;
  
    
    LefAnalyt(x,p) = (2 * p[4]/s) * atan.(2*(x .+ p[2]) ./ p[1]) .+ p[3]

    fit_result = curve_fit(LefAnalyt, x, L_error, [s,d,c,A])
    s1,d1,c1,A1 = coef(fit_result)

    println(d1)

    model(x,p) = p[4] ./ ((x.+ p[2]) .^ 2 .+ (p[1]/2)^2) 
    #.+ 0*p[3]/(x[end]-x[1])
    println(s1)
    println("Fit error: $(mse(fit_result))")
    
    return x,LefAnalyt(x,[s1,d1,c1,A1]),s1,model(x,[s1,d1,c1,A1])

end

function interpolate_previous(t, y::AbstractArray{T}, dt,t0=0;idxmax=2*10^9) where {T}
    # Find the number of samples needed for the desired frequency
    num_samples = min(Int(ceil((t[end] - t[1]) / dt)),idxmax)

    # Initialize an array to store the interpolated values
    println(num_samples)
    interpolated_y = Vector{T}(undef, num_samples)

    # Initialize indices and values for the previous data point
    idx = 1
    prev_y = y[1]

    prog = Progress(100,  desc= "Interpolation Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    i_mod = max(1,length(interpolated_y)÷100)

    for i in eachindex(interpolated_y)
        if i%i_mod == 0
            next!(prog)
        end
        # Calculate the desired time point
        desired_t = (i - 1) * dt + t0

        # Check if the desired time point is greater than the current time point
        while t[idx] < desired_t
            @inbounds prev_y = y[idx]
            idx += 1
        end

        # Use the previous data point for interpolation
        @inbounds interpolated_y[i] = prev_y
    end

    return interpolated_y
end


function interpolate_outcoupled(t, y::AbstractArray{T}, dt::Float64,t0=0;idxmax=2*10^9) where {T}
    # Find the number of samples needed for the desired frequency
    num_samples = min(Int(ceil((t[end] - t[1]) / dt)),idxmax)

    # Initialize an array to store the interpolated values
    println(num_samples)
    interpolated_y = Vector{T}(undef, num_samples)

    # Initialize indices and values for the previous data point
    idx = 1
    prev_y = y[1]

    prog = Progress(100, desc= "Interpolation Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    i_mod = length(interpolated_y)÷100

    for i in eachindex(interpolated_y)
        if i%i_mod == 0
            next!(prog)
        end
        # Calculate the desired time point
        desired_t = (i - 1) * dt + t0

        prev_y = 0.0+0.0*im
        # Check if the desired time point is greater than the current time point
        while t[idx] < desired_t
            @inbounds prev_y += y[idx]
            idx += 1
        end

        # Use the previous data point for interpolation
        @inbounds interpolated_y[i] = prev_y
    end

    return interpolated_y
end


function calc_spec(times,out::AbstractArray{T},durations::AbstractArray{Float64},freqs;mode=:sinc) where T<:Complex
    tmax = 1/(freqs[2]-freqs[1])
    println(tmax)
    spec = zeros(T,length(freqs))
    psd = zeros(Float64,length(freqs))
    prog = Progress(100, desc= "Spectrum Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    
    i_mod = length(out)÷100
    t_curr = 0
    
    if mode == :sinc
        update_spec! = update_spec_sinc!
    elseif mode == :lor
        update_spec! = update_spec_lor!
    elseif mode == :bare
        update_spec! = update_spec_bare!
    elseif mode == :norm
        update_spec! = update_spec_sinc_norm!
    else
        error("No valid mode provided. Choose between :sinc, :lor, :bare")
    end

    for j in eachindex(out)
        if j%i_mod == 0
            next!(prog)
        end
        for i in eachindex(freqs)
            update_spec!(spec,out,freqs,times,durations,i,j)
        end
        if times[j]-t_curr > tmax 
            psd += abs.(spec).^2
            spec .= 0
            t_curr = times[j]
        end
    end
    psd += abs.(spec).^2
    
    return psd
end

function update_spec_sinc!(spec,out,freqs,times,durations,i,j,shift=0)
    @inbounds spec[i] += 1/(2*pi)*Base.sinc((freqs[i]-shift)*durations[j]*2)*(durations[j]*2)*out[j]*exp(-2*pi*im*(freqs[i])*(times[j]))
end

function update_spec_lor!(spec,out,freqs,times,durations,i,j,shift=0)
    @inbounds spec[i] += -1/(2*pi)*1/(2*pi*im*(freqs[i]-shift)+1/durations[j]/2)*out[j]*exp(-2*pi*im*(freqs[i])*(times[j]))
end

function update_spec_bare!(spec,out,freqs,times,durations,i,j,shift=0)
    @inbounds spec[i] += out[j]*exp(-2*pi*im*freqs[i]*(times[j]))
end


function calc_spec(times,out::AbstractArray{T},durations::Float64,freqs;mode=:lor) where T<:Complex
    tmax = 1/(freqs[2]-freqs[1])
    println(tmax)
    spec = zeros(T,length(freqs))
    psd = zeros(Float64,length(freqs))
    prog = Progress(100, desc= "Spectrum Progress: ", barglyphs=BarGlyphs("[=> ]"),barlen=50)
    
    i_mod = length(out)÷100
    t_curr = 0
    
    if mode == :sinc
        update_spec! = update_spec_sinc!
    elseif mode == :lor
        update_spec! = update_spec_lor!
    elseif mode == :bare
        update_spec! = update_spec_bare!
    else
        error("No valid mode provided. Choose between :sinc, :lor, :bare")
    end

    for j in eachindex(out)
        if j%i_mod == 0
            next!(prog)
        end
        for i in eachindex(freqs)
            update_spec!(spec,out,freqs,times,durations,i,j)
            #@inbounds spec[i] += -1/(2*pi)*1/(2*pi*im*freqs[i]+1/(durations*2))*out[j]*exp(2*pi*im*freqs[i]*(times[j]))
            #@inbounds spec[i] += Base.sinc(freqs[i]*durations*2)*(durations[j]*2)*out[j]*exp(2*pi*im*freqs[i]*(times[j]))
        #
        end
        if times[j]-t_curr > tmax 
            psd += abs.(spec).^2
            spec .= 0
            t_curr = times[j]
        end
    end
    psd += abs.(spec).^2  
    return psd
end

