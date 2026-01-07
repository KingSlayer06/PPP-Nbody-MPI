# Include implementation
include("nbody-seq-src.jl")

function main()
    # Parse args
    verbose = false
    silent = false
    arg = 1
    while arg < length(ARGS) && startswith(ARGS[arg], "-")
        if ARGS[arg] == "-v"
            verbose = true
        elseif ARGS[arg] == "-s"
            silent = true
        else
            println("unknown option: ", ARGS[arg])
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
        exit(1)
    end

    # RUN
    for _ in 1:nruns
        perf_nbodies(nbodies,iterations,verbose,silent)
    end
end

main()
