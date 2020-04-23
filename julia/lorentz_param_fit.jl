module lorenz_model

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

model_params=[10.0,28.0,8.0/3]
all_ic=[[1.0,0.0,0.0],[0.5,1.0,0.0], [3,0.2,0.1]];
tend=10.0
tstart=0.0
sampling=0.05



function model(du,u,p,t)
    x,y,z = u
    σ,ρ,β = p
    du[1] = dx = σ*(y-x)
    du[2] = dy = x*(ρ-z) - y
    du[3] = dz = x*y - β*z
end


function InitPlot()

    mycolor=[:blue, :red, :green, :purple, :black, :cyan, :orange]
    t=tstart:sampling:100.0
    odedata=[ solve(ODEProblem(model,η,(tstart,100.0), model_params), Tsit5(), saveat=t, abstol=1e-8,reltol=1e-6) for η in all_ic];
    pl = plot(t,odedata[1][1,:],label="data 1", color=mycolor[1], lw=2, legend=false)

    for k in 2:length(all_ic)
        plot!(pl,t,odedata[k][1,:],label="data $k", color=mycolor[k], lw=2)
    end
    title!(pl, "Lorenz Chaos")
    xlabel!(pl, "Time")
    ylabel!(pl, "Variable")
    savefig("Lorenz_init.png")
    display(pl)

end


function predict_adjoint(param) # Our 1-layer neural network
    prob=ODEProblem(model,[1.0,0.0,0.0],(tstart,tend), model_params)
    Array(concrete_solve(prob,Tsit5(),param[1:3],param[4:end],saveat=tstart:sampling:tend,abstol=1e-8,reltol=1e-6))
end


# Generate some data to fit, and add some noise to it
data=predict_adjoint([1.0,0.0,0.0,10.0,28.0,8.0/3])
σN=0.05
data+=σN*randn(size(data))


#Init guess for the parameters
pguess=[0.8,0.1,0.1,10.5,28.5,9.0/3]


function loss_adjoint(param)
  prediction = predict_adjoint(param)
  loss = sum(abs2,prediction - data)
  loss
end

function train_model(;pguess=[1.1,0.05,-0.05,10.2,28.2,9.0/3])
    println("The initial loss function is $(loss_adjoint(pguess)[1])")
    resinit = DiffEqFlux.sciml_train(loss_adjoint,pguess,ADAM(), maxiters=4000)
    res = DiffEqFlux.sciml_train(loss_adjoint,resinit.minimizer,BFGS(initial_stepnorm=0.0001), maxiters=4000)
    println("The parameters are $(res.minimizer) with final loss value $(res.minimum)")
    return(res)

end

function plotFit(param; vars=(1,2,3), tend=100.0)

    validationPlot(param, param[1:3], vars=vars, tend=tend)

end

function validationPlot(param, ic; vars=(1,2,3), tend=100.0)

    sol_fit=solve(ODEProblem(model,ic,(0.0,tend),param[4:end]), Tsit5())
    sol_actual=solve(ODEProblem(model,ic,(0.0,tend),model_params), Tsit5())
    pl=plot(sol_fit, lw=2, legend=false, vars=vars, color=:green)
    plot!(pl,sol_actual,vars=vars,color=:blue)
    title!(pl,"Model Parameter Fits")
    savefig("valid_plot_lorenz2.png")
    display(pl)

end

end #module

lorenz_model.InitPlot()

resL=lorenz_model.train_model()

lorenz_model.plotFit(resL.minimizer; vars=(1,2,3), tend=30.0)

lorenz_model.plotFit(resL.minimizer; vars=(0,1), tend=30.0)
