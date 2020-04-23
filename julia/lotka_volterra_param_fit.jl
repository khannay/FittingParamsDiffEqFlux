#=
This first example comes almost straight from the documentation for DiffEqFlux

https://github.com/SciML/DiffEqFlux.jl#optimizing-parameters-of-an-ode-for-an-optimal-control-problem
https://julialang.org/blog/2019/01/fluxdiffeq/

=#



module lotka_volterra


using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

u0 = [1.0,1.0]
tstart=0.0
tend=10.0
sampling=0.1

model_params= [1.5,1.0,3.0,1.0]

function model(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

function InitPlot(;vars=(0,1))
    #Simple function to plot the dynamics
    tspan=(tstart,tend)
    prob = ODEProblem(model,u0,tspan,model_params)
    sol = solve(prob,Tsit5())
    plot(sol, lw=2, legend=false, vars=(0,1))
end

function predict_adjoint(param) # Our 1-layer neural network
    prob=ODEProblem(model,[1.0,0.0],(tstart,tend), model_params)
    Array(concrete_solve(prob,Tsit5(),param[1:2],param[3:end],saveat=tstart:sampling:tend, abstol=1e-8,reltol=1e-6))
end

# Generate some data to fit, and add some noise to it
data=predict_adjoint([1.0,1.0,1.5,1.0,3.0,1.0])
σN=0.1
data+=σN*randn(size(data))
data=abs.(data) #Keep measurements positive


# Returning more than just the loss function breaks the Flux optim
function loss_adjoint(param)
  prediction = predict_adjoint(param)
  loss = sum(abs2,prediction - data)
  loss
end;

#Test

function train_model(;pguess=[0.8,1.2,1.2,1.0,2.9,1.1])
    println("The initial loss function is $(loss_adjoint(pguess)[1])")
    #Train the ODE
    resinit=DiffEqFlux.sciml_train(loss_adjoint,pguess,ADAM(), maxiters=3000)
    res = DiffEqFlux.sciml_train(loss_adjoint,resinit.minimizer,BFGS(initial_stepnorm = 1e-5))
    println("The parameters are $(res.minimizer) with final loss value $(res.minimum)")
    return(res)
end


function plotFit(param)

    tspan=(tstart,tend)
    sol_fit=solve(ODEProblem(model,param[1:2],tspan,param[3:end]), Tsit5())

    tgrid=tstart:sampling:tend
    pl=plot(sol_fit, lw=2, legend=false)
    scatter!(pl,tgrid, data[1,:], color=:blue)
    scatter!(pl,tgrid, data[2,:], color=:red)
    xlabel!(pl,"Time")
    ylabel!(pl,"Population")
    title!(pl,"Model Parameter Fits")
    savefig("Lotka_Volterra_ParamFit.png")
    display(pl)
end

function validationPlot(param, ic)
    tspan=(tstart,tend)
    sol_fit=solve(ODEProblem(model,ic,tspan,param[3:end]), Tsit5())
    sol_actual=solve(ODEProblem(model,ic,tspan,model_params), Tsit5(), saveat=0.0:0.1:10.0)


    pl=scatter(sol_actual, lw=2.0, color=:blue, vars=(0,1))
    scatter!(sol_actual, color=:red, vars=(0,2))
    plot!(pl, sol_fit, lw=2, legend=false, color=:blue, vars=(0,1))
    plot!(pl, sol_fit, lw=2, color=:red, vars=(0,2))
    xlabel!(pl,"Rabbits")
    ylabel!(pl,"Lynx")
    title!(pl,"Validation Plot")
    savefig("Lotka_Volterra_Validation_Plot.png")
    display(pl)
end

end #module

lotka_volterra.InitPlot()

resL=lotka_volterra.train_model()

lotka_volterra.plotFit(resL.minimizer)

lotka_volterra.validationPlot(resL.minimizer, [6.,6.])
