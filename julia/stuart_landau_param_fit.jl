module sl_model

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

model_params=[0.1, 6.28, 0.1, 1.0];
all_ic=[[0.0,0.01], [π,0.2], [π/2.0, 0.1]];
tend=15.0
tstart=0.0
sampling=0.1



function sl_oscillator(du,u,p,t)
    γ, ω0, β, K=p
    ψ,R=u
    du[1]=ω0+K/2.0*sin(β)*(1+R^2)
    du[2]=-γ*R+K/2.0*cos(β)*R*(1-R^2)
end


function InitPlot()

    mycolor=[:blue, :red, :green, :purple, :black, :cyan, :orange]
    odedata=[solve(ODEProblem(sl_oscillator,η,(tstart,tend), model_params), Tsit5()) for η in all_ic];
    pl = plot(odedata[1],proj=:polar,vars=(1,2), color=:blue, legend=false)

    for k in 2:length(all_ic)
        plot!(pl, odedata[k], color=mycolor[k], vars=(1,2), proj=:polar)
    end
    title!(pl, "Oscillator Model")
    xlabel!(pl, "X")
    ylabel!(pl, "Y")
    display(pl)

end


function predict_adjoint(param) # Our 1-layer neural network
    prob=ODEProblem(sl_oscillator,[0.1,0.1],(tstart,tend), model_params)
    Array(concrete_solve(prob,Tsit5(),param[1:2],param[3:end],saveat=tstart:sampling:tend))
end


# Generate some data to fit, and add some noise to it
data=predict_adjoint([0.0,0.08,0.1, 6.28, 0.1, 1.0])
σN=0.05
data+=σN*randn(size(data))
data=abs.(data)




function loss_adjoint(param)
  prediction = predict_adjoint(param)
  loss = sum(abs2,prediction - data)
  loss,prediction
end

function train_model(;pguess=[0.1,0.1,0.15, 6.0, 0.2, 0.8])
    println("The initial loss function is $(loss_adjoint(pguess)[1])")
    res = DiffEqFlux.sciml_train(loss_adjoint,pguess,BFGS(initial_stepnorm = 0.0001))
    println("The parameters are $(res.minimizer) with final loss value $(res.minimum)")
    return(res)

end

function plotFit(param)

    validationPlot(param, param[1:2])

end

function validationPlot(param, ic)

    lay = @layout [ a{0.6w} [b; c]]

    sol_fit=solve(ODEProblem(sl_oscillator,ic,(0.0,15.0),param[3:end]), Tsit5(), saveat=tstart:0.01:tend)
    sol_data=solve(ODEProblem(sl_oscillator,ic,(0.0,15.0),model_params), Tsit5(), saveat=tstart:0.01:tend)
    pl1=plot(sol_fit,lw=2, legend=false, vars=(1,2), color=:green, proj = :polar)
    plot!(pl1, sol_data, color=:blue, proj=:polar, vars=(1,2))


    #radius plot
    pl2=plot(sol_fit,lw=2, legend=false, vars=(0,2), color=:green)
    plot!(pl2, sol_data, color=:blue, vars=(0,2), lw=2)

    #angle plot
    pl3=plot(sol_fit.t,sin.(sol_fit[1,:]), lw=2, legend=false, color=:green)
    plot!(pl3, sol_data.t, sin.(sol_data[1,:]), color=:blue)


    pl=plot(pl1, pl2, pl3, layout=lay)
    display(pl)

end

end #module

sl_model.InitPlot()
res=sl_model.train_model()
sl_model.plotFit(res.minimizer)
sl_model.validationPlot(res.minimizer, [π, 0.6])
