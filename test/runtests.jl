using PSMV
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here

n = 2*10^5

A = sprand(n, n, 0.05)
x = rand(n)
C = MultithreadedTransMatVec(A, Q)
@test norm(A*x - C*x) < 1e-10

C = MultithreadedMatVec(A)
@test norm(A*x - C*x) < 1e-10


