using PSMV
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here

n = 2*10^5

A = sprand(n, n, 0.0005)
x = rand(n)
C = PSMV.MultithreadedTransMatVec(A, A')
@show @test norm(A*x - C*x) < 1e-10

C = PSMV.MultithreadedMatVec(A, 1)
@show @test norm(A*x - C*x) < 1e-10

