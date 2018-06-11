module PSMV

# package code goes here

using Iterators
#simple_partition(n::Int, k::Int) = push!(map(first, partition(1:n,floor(Int,n/k))),n+1)
function simple_partition(n::Int,k::Int)
  rval = ones(Int,k+1)
  m,r = divrem(n,k)
  fill!(rval, m)
  rval[1] = 1
  for i = 2:k+1
    rval[i] += r .> (i-2)
  end
  cumsum!(rval,rval)
end

import Base.LinAlg.At_mul_B, Base.LinAlg.At_mul_B!, Base.LinAlg.A_mul_B!

"""
A type to encapsulate the multithreaded matrix-vector product operation.

Usage
-----
~~~~
A = sprandn(10^5,10^5,100/10^5); # create a fairly dense sparse matrix.
x = randn(10^5);
M = MultithreadedMatVec(A, simple_partition(A.n, Threads.nthreads()));
y = M.'*x;
norm(y - A.'*x)
~~~~
"""
mutable struct MultithreadedMatVec{T,I} <: AbstractArray{T,2}
  A::SparseMatrixCSC{T,I}
  regions::Vector{Int}

  MultithreadedMatVec(A::SparseMatrixCSC) = MultithreadedMatVec(A,Threads.nthreads())
  MultithreadedMatVec(A::SparseMatrixCSC, k::Int) = MultithreadedMatVec(A, simple_partition(A.n,k))
  function MultithreadedMatVec(A::SparseMatrixCSC{T,I}, regions::Vector{Int}) where {T,I}
    new{T,I}(A, regions)
  end
end

import Base.size
size(M::MultithreadedMatVec,inputs...) = size(M.A,inputs...)

# Julia's built in multiplication operations are called with
# A_mul_B!
# Ac_mul_B!
# At_mul_B!
# which take in
# α::Number, A::SparseMatrixCSC, B::StridedVecOrMat, β::Number, C::StridedVecOrMat
# and compute
# βC += αA B
# βC += αA^* B
# βC += αA^T B
# respectively
# look in base/sparse/linalg.jl
# for their implementations

""" Run the internal loop """
function internal_loop_transmult(C,B,nzv,rv,colptr,i1,i2,α::Number)
  for k=1:size(C,2)
    @inbounds for col=i1:i2
      tmp = zero(eltype(C))
      for j=colptr[col]:(colptr[col+1]-1)
        tmp += nzv[j]*B[rv[j],k]
      end
      C[col,k] += α*tmp
    end
  end
  return
end

"""
we are going to make these work with MuilthreadedMatVec types
"""
function At_mul_B!(α::Number, M::MultithreadedMatVec, B::StridedVecOrMat, β::Number, C::StridedVecOrMat)
  M.A.m == size(B,1) || throw(DimensionMismatch())
  M.A.n == size(C,1) || throw(DimensionMismatch())
  size(B,2) == size(C,2) || throw(DimensionMismatch())
  nzv = M.A.nzval
  rv = M.A.rowval
  colptr = M.A.colptr
  if β != 1
    β != 0 ? scale!(C, β) : fill!(C, zero(eltype(C)))
  end
  # this is the parallel construction
  Threads.@threads for t=1:(length(M.regions)-1)
    internal_loop_transmult(C,B,nzv,rv,colptr,M.regions[t],M.regions[t+1]-1,α)
  end
  C
end

function At_mul_B(M::MultithreadedMatVec{TA,S}, x::StridedVector{Tx}) where {TA,S,Tx}
  T = promote_type(TA,Tx)
  At_mul_B!(one(T), M, x, zero(T), similar(x, T, M.A.n))
end

module MyAtomics

lt = "double" # llvmtype
ilt = "i64" # llvmtype of int representation

import Base.Sys: ARCH, WORD_SIZE

@inline function atomic_cas!(x::Array{Float64}, ind::Int, oldval::Float64, newval::Float64)
    xptr = Base.unsafe_convert(Ptr{Float64}, x)
    xptr += 8*(ind-1)
    # on new versions of Julia, this should be
    # %iptr = inttoptr i$WORD_SIZE %0 to $ilt*
    Base.llvmcall( """%iptr = bitcast $lt* %0 to $ilt*
                      %icmp = bitcast $lt %1 to $ilt
                      %inew = bitcast $lt %2 to $ilt
                      %irs = cmpxchg $ilt* %iptr, $ilt %icmp, $ilt %inew acq_rel acquire
                      %irv = extractvalue { $ilt, i1 } %irs, 0
                      %rv = bitcast $ilt %irv to $lt
                      ret $lt %rv
                """,
    Float64,  # return type
    Tuple{Ptr{Float64}, Float64, Float64},  # argument types
    xptr, oldval, newval # arguments
    )
end

@inline function atomic_add!(x::Array{Float64}, ind::Int, val::Float64)
  IT = Int64
  xptr = Base.unsafe_convert(Ptr{Float64}, x)
  xptr += 8*(ind-1)

  while true
    oldval = x[ind]
    newval = oldval + val

    #old2 = atomic_cas!(x, ind, oldval, newval)
    # inline this call and optimize out some stuff
    old2 = Base.llvmcall( """%iptr = bitcast $lt* %0 to $ilt*
                      %icmp = bitcast $lt %1 to $ilt
                      %inew = bitcast $lt %2 to $ilt
                      %irs = cmpxchg $ilt* %iptr, $ilt %icmp, $ilt %inew acq_rel acquire
                      %irv = extractvalue { $ilt, i1 } %irs, 0
                      %rv = bitcast $ilt %irv to $lt
                      ret $lt %rv
                """,
    Float64,  # return type
    Tuple{Ptr{Float64}, Float64, Float64},  # argument types
    xptr, oldval, newval # arguments
    )

    if reinterpret(IT,oldval) == reinterpret(IT,old2)
      return newval
    end
  end
end
end

function internal_loop_mult(C,B,nzv,rv,colptr,i1,i2,α::Number)
  for k=1:size(C,2)
    koffset = size(C,1)*(k-1)
    @inbounds for col=i1:i2
      αxj = α*B[col,k]
      for j=colptr[col]:(colptr[col+1]-1)
        # need to be done atomically...
        #Threads.atomic_add!(C[rv[j],k], nzv[j]*αxj)
        MyAtomics.atomic_add!(C,koffset+rv[j],nzv[j]*αxj)
      end
    end
  end
  return
end

function A_mul_B!(α::Number, M::MultithreadedMatVec, B::StridedVecOrMat, β::Number, C::StridedVecOrMat)
  M.A.n == size(B,1) || throw(DimensionMismatch())
  M.A.m == size(C,1) || throw(DimensionMismatch())
  size(B,2) == size(C,2) || throw(DimensionMismatch())
  nzv = M.A.nzval
  rv = M.A.rowval
  colptr = M.A.colptr
  if β != 1
    β != 0 ? scale!(C, β) : fill!(C, zero(eltype(C)))
  end
  # this is the parallel construction
  Threads.@threads for t=1:(length(M.regions)-1)
    internal_loop_mult(C,B,nzv,rv,colptr,M.regions[t],M.regions[t+1]-1,α)
  end
  C
end

import Base.eltype
eltype(M::MultithreadedMatVec{T,I}) where {T,I} = T

function A_mul_B!(C::StridedVecOrMat, M::MultithreadedMatVec, B::StridedVecOrMat)
  T = promote_type(eltype(M), eltype(B))
  A_mul_B!(one(T), M, B, zero(T), C)
end



mutable struct MultithreadedTransMatVec{T,I} <: AbstractArray{T,2}
  A::SparseMatrixCSC{T,I}
  Q::SparseMatrixCSC{T,I}
  regions::Vector{Int}


  function MultithreadedTransMatVec(A::SparseMatrixCSC{T,I}, Q::SparseMatrixCSC, regions::Vector{Int}) where {T,I}
    new{T,I}(A, Q, regions)
  end

end


function MultithreadedTransMatVec(A::SparseMatrixCSC, Q::SparseMatrixCSC, k::Int64)
  MultithreadedTransMatVec(A, Q, simple_partition(A.n,k))
end

function MultithreadedTransMatVec(A::SparseMatrixCSC, Q::SparseMatrixCSC)
  MultithreadedTransMatVec(A,Q,Threads.nthreads())
end



size(M::MultithreadedTransMatVec,inputs...) = size(M.A,inputs...)


 

function internal_loop_mult_opt(C,B,nzv,rv,colptr,i1,i2,α::Number)
  k=1
  #for k=1:size(C,1)
    @inbounds for col=i1:i2
      tmp = zero(eltype(C))
      for j=colptr[col]:(colptr[col+1]-1)
        tmp += nzv[j]*B[rv[j],k]
      end
      C[col,k] += α*tmp
    end
  #end
  return
end

function A_mul_B!(α::Number, M::MultithreadedTransMatVec, B::StridedVector, β::Number, C::StridedVector)
  M.A.n == size(B,1) || throw(DimensionMismatch())
  M.A.m == size(C,1) || throw(DimensionMismatch())
  size(B,2) == size(C,2) || throw(DimensionMismatch())
  nzv = M.Q.nzval
  rv = M.Q.rowval
  colptr = M.Q.colptr

  if β != 1
    β != 0 ? scale!(C, β) : fill!(C, zero(eltype(C)))
  end

  # this is the parallel construction
  Threads.@threads for t=1:(length(M.regions)-1)
    internal_loop_mult_opt(C,B,nzv,rv,colptr,M.regions[t],M.regions[t+1]-1,α)
  end
  C
end

import Base.eltype
eltype(M::MultithreadedTransMatVec{T,I}) where {T,I} = T

function A_mul_B!(C::StridedVector, M::MultithreadedTransMatVec, B::StridedVector)
  T = promote_type(eltype(M), eltype(B))
  A_mul_B!(one(T), M, B, zero(T), C)
end



function test_perf()
  n = 2*10^4
  println("Constructing Sparse Matrix")
  @time A = sprandn(n,n,100/n); # create a fairly dense sparse matrix.
  @time Q = A'

  x = randn(n);
  M = MultithreadedMatVec(A, simple_partition(n, 1));
  @show norm(M*x - A*x)
  println("Ax")
  @time A*x;
  @time A*x;
  @time A*x;

  println("Mx")
  @time M*x;
  @time M*x;
  @time M*x;

  M = MultithreadedMatVec(A, simple_partition(n, 2));
  println("Mx - 2")
  @show norm(M*x - A*x)
  @time M*x;
  @time M*x;
  @time M*x;

  M = MultithreadedMatVec(A, simple_partition(n, 4));
  println("Mx - 4")
  @show norm(M*x - A*x)
  @time M*x;
  @time M*x;
  @time M*x;

  M = MultithreadedMatVec(A, simple_partition(n, 6));
  println("Mx - 6")
  @show norm(M*x - A*x)
  @time M*x;
  @time M*x;
  @time M*x;

  M = MultithreadedMatVec(A, simple_partition(n, 12));
  println("Mx - 12")
  @show norm(M*x - A*x)
  @time M*x;
  @time M*x;
  @time M*x;

  println("My benchmarks")
  @time C = MultithreadedTransMatVec(A, Q, 1)
  @time @show norm(C*x - A*x)
  println("Ax")
  @time A*x
  @time A*x
  @time A*x

  println("Cx")
  @time C*x
  @time C*x
  @time C*x


  C = MultithreadedTransMatVec(A, Q, 2)
  println("Cx - 2")
  @time C*x
  @time C*x
  @time C*x


  C = MultithreadedTransMatVec(A, Q, 4)
  println("Cx - 4")
  @time C*x
  @time C*x
  @time C*x


  C = MultithreadedTransMatVec(A, Q, 8)
  println("Cx - 8")
  @time C*x
  @time C*x
  @time C*x

  C = MultithreadedTransMatVec(A, Q, 16)
  println("Cx - 16")
  @time C*x
  @time C*x
  @time C*x

  C = MultithreadedTransMatVec(A, Q)
  println("Cx - ?")
  @time C*x
  @time C*x
  @time C*x





  return 0
end



end # module
