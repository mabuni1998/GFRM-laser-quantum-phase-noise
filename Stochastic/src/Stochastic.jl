module Stochastic
    using Random
    using ProgressMeter

    export StochasticRateEquation,
    gfrm_out,gfrm_intra,gfrm,
    RIN,interpolate_previous,interpolate_outcoupled,fit_lorr,fit_lorr_cum,
    calc_spec

    include("GFRM.jl")
    include("SignalProcessing.jl")

end # module Stochastic
