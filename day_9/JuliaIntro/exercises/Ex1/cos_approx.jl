using BenchmarkTools

function cos_approx(x, N)
    a= 0
    for j in 0:N
        V=(((-1)^j) /(factorial(big(2*j))) ) * (x^(2j))
        a += V
    end
    return a
    # approximation of cosine via power series expansion
    # inputs:
    #       - x : argument of cosine 
    #       - N : truncation order of the power series approximation
    # outputs:
    #       - cos_val : approximation of cos(x)
end

@btime cos_approx($(π/3),$(10)) 
@btime cos($(π/3))
@btime cos(π/3)