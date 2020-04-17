using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots


function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5())
plot(sol, lw=2, legend=false, vars=(0,1))

function predict_adjoint(param) # Our 1-layer neural network
    Array(concrete_solve(prob,Tsit5(),param[1:2],param[3:end],saveat=0.0:0.1:10.0))
end

# Generate some data to fit, and add some noise to it
data=predict_adjoint([1.0,1.0,1.5,1.0,3.0,1.0])
σN=0.1
data+=σN*randn(size(data))
data=abs.(data) #Keep measurements positive

#Init guess for the parameters
pguess=[0.8,1.2,1.2,1.0,2.9,1.1]


function loss_adjoint(param)
  prediction = predict_adjoint(param)
  loss = sum(abs2,prediction - data)
  loss,prediction
end;

#Test

println("The initial loss function is $(loss_adjoint(pguess)[1])")


#Train the ODE
res = DiffEqFlux.sciml_train(loss_adjoint,pguess,BFGS(initial_stepnorm = 0.00001))

println("The parameters are $(res.minimizer) with final loss value $(res.minimum)")


function plotFit(param)

    sol_fit=solve(ODEProblem(lotka_volterra,param[1:2],tspan,param[3:end]), Tsit5())


    pl=plot(sol_fit, lw=2, legend=false)
    scatter!(pl,0.0:0.1:10.0, data[1,:], color=:blue)
    scatter!(pl, 0.0:0.1:10.0, data[2,:], color=:red)
    xlabel!(pl,"Time")
    ylabel!(pl,"Population")
    title!(pl,"Model Parameter Fits")
    display(pl)
end

plotFit(res.minimizer)
