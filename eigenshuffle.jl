using LinearAlgebra
using Hungarian
using LazyGrids: ndgrid


function distancematrix(vec1, vec2)
    """simple interpoint distance matrix"""
    vec1, vec2 = ndgrid(vec1, vec2)
    return abs.(vec1 - vec2)
end


function outerplus(M, x, y)
    nx, ny = size[M]
    minval = Inf
    for r = 1:nx
        x1 = x[r]
        for c = 1:ny
            M[r,c] = M[r,c] - (x1 + y[c])
            if minval > M[r,c]
                minval = M[r,c]
            end
        end
    end
    i1 = findfirst(M .== minval)
    rIdx = i1[1]
    cIdx = i1[2]
    return minval, rIdx, cIdx
end


function eigenshuffle(Asequence, Bsequence=nothing)
    """Consistent sorting for an eigenvalue/vector sequence.
    
    This is a translation of the Matlab code
    http://www.mathworks.com/matlabcentral/fileexchange/20652

    % Arguments: (input)
    %  Asequence - an array of eigenvalue problems. If
    %      Asequence is a 3-d numeric array, then each
    %      plane of Asequence must contain a square
    %      matrix that will be used to call eig.
    %
    %      Eig will be called on each of these matrices
    %      to produce a series of eigenvalues/vectors,
    %      one such set for each eigenvalue problem.
    %
    %  Bsequence - (OPTIONAL) This allows the user to solve
    %      generalized eigenvalue problems of the form
    %
    %         A*X = lambda*B*x
    %
    %      Bsequence must be the same shape and size as
    %      Asequence.
    %
    %      ONLY supply Bsequence when generalized eigenvalues
    %      will be computed.
    %
    % Arguments: (Output)
    %  Vseq - a 3-d array (pxpxn) of eigenvectors. Each
    %      plane of the array will be sorted into a
    %      consistent order with the other eigenvalue
    %      problems. The ordering chosen will be one
    %      that maximizes the energy of the consecutive
    %      eigensystems relative to each other.
    %
    %  Dseq - pxn array of eigen values, sorted in order
    %      to be consistent with each other and with the
    %      eigenvectors in Vseq.
    """
    # Is Asequence a 3-d array?
    Asize = size(Asequence)
    if (Asize[1] != Asize[2])
        error("Asequence must be a (pxpxn) array of eigen-problems, each of size pxp")
    end
    
    # was Bsequence supplied
    genflag = false
    if !isnothing(Bsequence) && !isempty(Bsequence)
      genflag = true
      if !isequal(size(Asequence), size(Bsequence)) 
        error("If Bsequence is provided to compute generalized eigenvalues, then Asequence and Bsequence must be the same size arrays")
      end
    end
    
    p = Asize[1]
    if length(Asize) < 3
      n = 1
    else
      n = Asize[3]
    end
    
    # the initial eigenvalues/vectors in nominal order
    Vseq = zeros(ComplexF64, p,p,n)
    Dseq = zeros(ComplexF64, p,n)
    for i = 1:n
        if genflag
            D, V = eigen(Asequence[:,:,i], Bsequence[:,:,i])
        else
            D, V = eigen(Asequence[:,:,i])
        end
        # initial ordering is purely in decreasing order.
        # If any are complex, the sort is in terms of the real part.
        junk = sort(real.(D); rev=true)
        tags = [findfirst(isapprox.(junk, real(D[i]))) for i in eachindex(D)]
        
        Dseq[:,i] = D[tags]
        Vseq[:,:,i] = V[:,tags]
    end
    
    # was there only one eigenvalue problem?
    if n < 2
        # we can quit now, having sorted the eigenvalues as best as we could.
        return Vseq, Dseq
    end
    
    # now, treat each eigenproblem in sequence (after the first one.)
    for i = 2:n
        # compute distance between systems
        V1 = Vseq[:,:,i-1]
        V2 = Vseq[:,:,i]
        D1 = Dseq[:,i-1]
        D2 = Dseq[:,i]
        dist = (1 .- abs.(V1' * V2)) .* sqrt.(
            distancematrix(real.(D1), real.(D2)) .^ 2 .+ 
            distancematrix(imag.(D1), imag.(D2)) .^ 2)
        
        # Is there a best permutation? use munkres.
        #reorder = munkres(dist)
        reorder, cost = hungarian(dist)
        
        Vseq[:,:,i] = Vseq[:,reorder,i]
        Dseq[:,i] = Dseq[reorder,i]
        
        # also ensure the signs of each eigenvector pair were consistent if possible
        S = real.(sum(Vseq[:,:,i-1] .* Vseq[:,:,i], dims=1)) .< 0
        for k in eachindex(S)
            if S[k]
                Vseq[:,k,i] = -Vseq[:,k,i]
            end
        end
    end
    return Vseq, Dseq
end

# %% Test
begin
    Efun(t) = [1      2*t+1  t^2    t^3;
            2*t+1  2-t    t^2    1-t^3;
            t^2    t^2    3-2*t  t^2;
            t^3    1-t^3  t^2    4-3*t]

    Aseq = zeros(4,4,21)
    for i = 1:21
        Aseq[:,:,i] = Efun((i-11)/10);
    end
    t = (-1:.1:1)

    Vseq, Dseq = eigenshuffle(Aseq)

    ref = [
        8.45350243849333	7.81207039358517	7.24814297493824	6.75243317420137	6.31556044512256	5.92825455803923	5.58164311623333	5.26758937520702	4.97909503023966	4.71089289265541	4.46050487001876	4.23020988479121	4.03026597357252	3.88170516735387	3.81080309483138	3.83021772733133	3.93014048244672	4.09266957505824	4.30423002544511	4.55722980801077	4.84821707758949
        5.00000000000000	4.76869965195071	4.56000538864125	4.36481455149044	4.17511785380649	3.98546097896941	3.79314407714438	3.59759343924863	3.39948847880225	3.19996664198450	3	2.79996980905802	2.59973106523242	2.40465558735811	2.14640989305628	1.89856712411017	1.59374301177233	1.23083724924163	0.825152817361920	0.403893140154046	-5.30803949273567e-16
        2.34468977469737	2.37278552990728	2.34131394305002	2.27094976834765	2.18568004614456	2.11183493715857	2.07267152021895	2.07681964608434	2.11560229943621	2.17419889144752	2.23912327825656	2.29713163669636	2.33034363135999	2.30635749157776	2.26280145660854	2.11113816690052	1.92977084833395	1.74502602460044	1.57290562018629	1.42718320659128	1.32732730976101
        0.201807786809296	0.446444424556846	0.650537693370494	0.811802505960530	0.923641654926389	0.974449525832792	0.952541286403351	0.857997539460009	0.705814191521879	0.514941573912567	0.300371851724682	0.0726886694544160	-0.160340670164937	-0.392718246289732	-0.620014444496206	-0.839923018342019	-1.05365434255300	-1.26853284890031	-1.50228846299332	-1.78830615475609	-2.17554438735050
    ]
    @assert all(isapprox(Dseq, ref))
end
