module kuramoto_model

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots, Random

rng=MersenneTwister(123) #make the results consistent across runs
N=10000
ω=randn(rng,N)
norder=2 #OA=1, versus Hannay Ansatz=2

#ω0,γ,K,β
model_params=[2π, 0.1,0.8,0.1]
ic=rand(rng,N)*2π;

tend=20.0
tstart=0.0
sampling=0.05


function full_model(dϕ,ϕ,p,t)
    ω0, γ, K, β=p
    ψ,R=OrderParameter(ϕ)
    for i in 1:N
        dϕ[i]=ω0+γ*ω[i]+R*K*sin(ψ-ϕ[i]+β)
    end
end

function reduced_model(du,u,p,t)
    ω0, γ, K, β=p
    ψ,R=u
    n=norder
    du[1]=ω0+K/2.0*sin(β)*(1+R^(2*norder))
    du[2]=-γ*R+K/2.0*cos(β)*R*(1-R^(2*norder))
end


function OrderParameter(ϕ;m=1.0)
    Z=1.0/N*sum(exp.(im.*ϕ*m))
    R=abs(Z)
    ψ=angle(Z)
    return ψ,R
end

function OrderParameterComplex(ϕ;m=1.0)
    Z=1.0/N*sum(exp.(im.*ϕ*m))
    return Z
end

icr=[angle(OrderParameterComplex(ic)), abs(OrderParameterComplex(ic))]

function InitPlot()

    lay = @layout [ a{0.6w} [b; c]]

    t=tstart:0.01:tend
    sol=solve(ODEProblem(full_model,ic,(tstart,tend),model_params), Tsit5(), saveat=t)

    num_tps=length(t)
    R=zeros(num_tps)
    ψ=zeros(num_tps)
    for tp in 1:num_tps
        psi, rr=OrderParameter(sol[:,tp])
        R[tp]=rr
        ψ[tp]=psi
    end

    pl1=plot(ψ,R,lw=2, legend=false, color=:blue, proj = :polar)


    #radius plot
    pl2=plot(t,R,lw=2, legend=false, color=:blue)


    #angle plot
    pl3=plot(t,sin.(ψ), lw=2, legend=false, color=:blue)


    pl=plot(pl1, pl2, pl3, layout=lay)
    display(pl)

end


function predict_adjoint(param) # Our 1-layer neural network
    prob=ODEProblem(reduced_model,icr,(tstart,tend), model_params)
    Array(concrete_solve(prob,Tsit5(),icr,param,saveat=tstart:sampling:tend,abstol=1e-8,reltol=1e-6))
end

function generate_data(ic; param=model_params)
    prob=ODEProblem(full_model,ic,(tstart,tend), param)
    tps=tstart:sampling:tend
    num_tps=length(tps)
    d1=Array(concrete_solve(prob,Tsit5(),ic,param,saveat=tps))
    # Get a time series of the order parameter
    d2=mapslices(OrderParameterComplex, d1, dims=1)
    data=Float64[angle.(d2); abs.(d2)]
    data=reshape(data,2,num_tps)
    return data
end

# Generate some data to fit, and add some noise to it
data=generate_data(ic)
#σN=0.05
#data+=σN*randn(size(data))


function loss_adjoint(param)
    prediction = predict_adjoint(param)
    loss=0.0
    for k in 1:length(prediction[1,:])
        Rp=prediction[2,k]
        Rd=data[2,k]
        θp=prediction[1,k]
        θd=data[1,k]
        loss+=sqrt((Rp*cos(θp)-Rd*cos(θd))^2+(Rp*sin(θp)-Rd*sin(θd))^2)
    end
    loss,prediction
end

function train_model(;pguess=[6.1, 0.15,1.1,0.15])
    println("The initial loss function is $(loss_adjoint(pguess)[1])")
    res = DiffEqFlux.sciml_train(loss_adjoint,pguess,BFGS(initial_stepnorm=0.0001), maxiters=4000)
    println("The parameters are $(res.minimizer) with final loss value $(res.minimum)")
    return(res)
end

function plotFit(param; tend=20.0)

    validationPlot(param,ic; tend=tend)

end

function validationPlot(param,ic; tend=10.0)

    lay = @layout [ a{0.6w} [b; c]]

    t=tstart:0.01:tend
    icr=[angle(OrderParameterComplex(ic)), abs(OrderParameterComplex(ic))]

    sol=solve(ODEProblem(full_model,ic,(tstart,tend),model_params), Tsit5(), saveat=t)
    sol_fit=solve(ODEProblem(reduced_model,icr,(tstart,tend),param), Tsit5(), saveat=t)

    num_tps=length(t)
    R=zeros(num_tps)
    ψ=zeros(num_tps)
    for tp in 1:num_tps
        psi, rr=OrderParameter(sol[:,tp])
        R[tp]=rr
        ψ[tp]=psi
    end

    pl1=plot(ψ,R,lw=2, legend=false, color=:blue, proj = :polar)
    plot!(pl1, sol_fit, color=:green, proj=:polar, vars=(1,2))

    #radius plot
    pl2=plot(t,R,lw=2, legend=false, color=:blue)
    plot!(pl2, sol_fit, color=:green, lw=2.0, vars=(0,2))

    #angle plot
    pl3=plot(t,sin.(ψ), lw=2, legend=false, color=:blue)
    plot!(pl3, sol_fit.t, sin.(sol_fit[1,:]), color=:green)

    pl=plot(pl1, pl2, pl3, layout=lay)
    display(pl)

end

end #module

#d=kuramoto_model.generate_data(randn(kuramoto_model.N))
#kuramoto_model.predict_adjoint(kuramoto_model.model_params)

kuramoto_model.InitPlot()
results_kuramoto=kuramoto_model.train_model()
kuramoto_model.plotFit(results_kuramoto.minimizer)
kuramoto_model.validationPlot(results_kuramoto.minimizer, π*rand(kuramoto_model.N))
kuramoto_model.validationPlot(results_kuramoto.minimizer, 1.5*π*rand(kuramoto_model.N))
