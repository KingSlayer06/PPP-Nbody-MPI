using MPI

# Include implementation
include("nbody-par-src.jl")

function main()
    # Initialize MPI
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Parse args
    verbose = false
    silent = false
    balance = false
    gather = false
    arg = 1
    while arg < length(ARGS) && startswith(ARGS[arg], "-")
        if ARGS[arg] == "-v"
            verbose = true
        elseif ARGS[arg] == "-s"
            silent = true
        elseif ARGS[arg] == "-balance"
            balance = true
        elseif ARGS[arg] == "-gather"
            gather = true
        else
            println("unknown option: ", ARGS[arg])
            MPI.Finalize()
            exit(1)
        end
        arg += 1
    end

    if length(ARGS) - arg == 2
        nbodies = parse(Int,ARGS[arg])
        iterations = parse(Int,ARGS[arg + 1])
        nruns = parse(Int,ARGS[arg + 2])
    else
        println("usage: nbody <nbodies> <iterations> <nruns>")
        MPI.Finalize()
        exit(1)
    end

    # RUN
    for _ in 1:nruns
        perf_nbodies(nbodies,iterations,comm,verbose,silent,balance,gather)
    end

    # Finalize MPI
    MPI.Finalize()
end

main()