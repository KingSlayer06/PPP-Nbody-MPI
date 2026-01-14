#!/usr/bin/env julia
# Benchmarking script for N-body simulation
# Tests different problem sizes and number of processes
#
# Usage:
#   julia benchmark.jl
#   or
#   ./benchmark.jl
#
# Note: Requires mpiexecjl or mpirun to be available

using Printf
using Statistics

function run_benchmark(nbodies, iterations, nprocs, silent=true)
    output_file = "bench_output_$(nprocs)_$(nbodies).txt"
    
    # Build command
    cmd_args = silent ? ["-s", string(nbodies), string(iterations), "1"] : 
                        [string(nbodies), string(iterations), "1"]
    
    try
        cmd = `mpiexecjl -np $nprocs julia -O3 nbody-par-run.jl $cmd_args`
        run(pipeline(cmd, stdout=output_file, stderr=devnull))
    catch
        # Fallback to mpirun
        cmd = `mpirun -np $nprocs julia -O3 nbody-par-run.jl $cmd_args`
        run(pipeline(cmd, stdout=output_file, stderr=devnull))
    end
    
    # Parse timing from output
    if isfile(output_file)
        output = read(output_file, String)
        rm(output_file)
        
        # Extract time: "nbody took X.XXX secs"
        for line in split(output, '\n')
            if startswith(line, "nbody took")
                time_str = match(r"(\d+\.\d+)", line)
                if time_str !== nothing
                    return parse(Float64, time_str.captures[1])
                end
            end
        end
    end
    
    return nothing
end

function run_sequential_benchmark(nbodies, iterations)
    output_file = "seq_bench_output.txt"
    cmd = `julia -O3 nbody-seq-run.jl -s $nbodies $iterations 1`
    run(pipeline(cmd, stdout=output_file, stderr=devnull))
    
    if isfile(output_file)
        output = read(output_file, String)
        rm(output_file)
        
        for line in split(output, '\n')
            if startswith(line, "nbody took")
                time_str = match(r"(\d+\.\d+)", line)
                if time_str !== nothing
                    return parse(Float64, time_str.captures[1])
                end
            end
        end
    end
    
    return nothing
end

function benchmark_config(nbodies, iterations, nprocs, nruns=1)
    times = Float64[]
    
    @printf("  Running %d process(es)... ", nprocs)
    for run in 1:nruns
        time = run_benchmark(nbodies, iterations, nprocs)
        if time !== nothing
            push!(times, time)
        end
    end
    
    if isempty(times)
        @printf("FAILED\n")
        return nothing
    end
    
    avg_time = mean(times)
    if nruns > 1
        @printf("%.3f secs (avg of %d runs)\n", avg_time, length(times))
    else
        @printf("%.3f secs\n", avg_time)
    end
    return avg_time
end

function benchmark_sequential(nbodies, iterations, nruns=1)
    times = Float64[]
    
    @printf("  Running sequential... ")
    for run in 1:nruns
        time = run_sequential_benchmark(nbodies, iterations)
        if time !== nothing
            push!(times, time)
        end
    end
    
    if isempty(times)
        @printf("FAILED\n")
        return nothing
    end
    
    avg_time = mean(times)
    if nruns > 1
        @printf("%.3f secs (avg of %d runs)\n", avg_time, length(times))
    else
        @printf("%.3f secs\n", avg_time)
    end
    return avg_time
end

function print_results_table(results)
    println()
    println("=" ^ 80)
    println("Benchmark Results")
    println("=" ^ 80)
    println()
    
    # Header
    @printf("%-12s", "Bodies")
    @printf("%-12s", "Iterations")
    procs_list = sort(unique([r[3] for r in results]))
    for nprocs in procs_list
        @printf("%-15s", "$nprocs procs")
    end
    @printf("%-15s", "Sequential")
    @printf("%-15s", "Best Speedup")
    println()
    println("-" ^ 80)
    
    # Group by (nbodies, iterations)
    configs = unique([(r[1], r[2]) for r in results])
    
    for (nbodies, iterations) in configs
        config_results = [r for r in results if r[1] == nbodies && r[2] == iterations]
        
        @printf("%-12d", nbodies)
        @printf("%-12d", iterations)
        
        seq_time = nothing
        times_by_proc = Dict()
        
        for (n, it, p, t) in config_results
            if p == 0
                seq_time = t
            else
                times_by_proc[p] = t
            end
        end
        
        # Print times for each process count
        for nprocs in procs_list
            if haskey(times_by_proc, nprocs)
                @printf("%-15.3f", times_by_proc[nprocs])
            else
                @printf("%-15s", "N/A")
            end
        end
        
        # Print sequential time
        if seq_time !== nothing
            @printf("%-15.3f", seq_time)
        else
            @printf("%-15s", "N/A")
        end
        
        # Calculate best speedup
        if seq_time !== nothing && !isempty(times_by_proc)
            best_par_time = minimum(values(times_by_proc))
            speedup = seq_time / best_par_time
            @printf("%-15.2fx", speedup)
        else
            @printf("%-15s", "N/A")
        end
        
        println()
    end
    
    println()
    println("=" ^ 80)
    println()
end

function print_efficiency_analysis(results)
    println("Efficiency Analysis")
    println("=" ^ 80)
    println()
    
    configs = unique([(r[1], r[2]) for r in results])
    
    for (nbodies, iterations) in configs
        config_results = [r for r in results if r[1] == nbodies && r[2] == iterations]
        
        seq_time = nothing
        for (n, it, p, t) in config_results
            if p == 0
                seq_time = t
                break
            end
        end
        
        if seq_time === nothing
            continue
        end
        
        @printf("Configuration: %d bodies, %d iterations\n", nbodies, iterations)
        @printf("  Sequential time: %.3f secs\n", seq_time)
        @printf("  Parallel results:\n")
        
        par_results = [(p, t) for (n, it, p, t) in config_results if p > 0]
        sort!(par_results, by=x -> x[1])
        
        for (nprocs, par_time) in par_results
            speedup = seq_time / par_time
            efficiency = speedup / nprocs * 100
            @printf("    %d processes: %.3f secs, speedup=%.2fx, efficiency=%.1f%%\n", 
                    nprocs, par_time, speedup, efficiency)
        end
        println()
    end
end

function save_results_to_csv(results, filename="benchmark_results.csv")
    # Open file for writing
    open(filename, "w") do io
        # Write header - processes first, then other parameters, time before speedup
        println(io, "processes,bodies,iterations,time,speedup,efficiency")
        
        # Group results by configuration to find sequential time for each
        configs = unique([(r[1], r[2]) for r in results])
        
        for (nbodies, iterations) in configs
            config_results = [r for r in results if r[1] == nbodies && r[2] == iterations]
            
            # Find sequential time for this configuration
            seq_time = nothing
            for (n, it, p, t) in config_results
                if p == 0
                    seq_time = t
                    break
                end
            end
            
            # Skip sequential result (process 0) - only write parallel results
            # Write parallel results
            par_results = [(p, t) for (n, it, p, t) in config_results if p > 0]
            sort!(par_results, by=x -> x[1])
            
            for (nprocs, par_time) in par_results
                if seq_time !== nothing
                    speedup = seq_time / par_time
                    efficiency = speedup / nprocs * 100
                else
                    speedup = 0.0
                    efficiency = 0.0
                end
                # Round speedup and efficiency to 2 decimal places
                speedup_rounded = round(speedup, digits=2)
                efficiency_rounded = round(efficiency, digits=2)
                time_rounded = round(par_time, digits=2)
                println(io, "$nprocs,$nbodies,$iterations,$time_rounded,$speedup_rounded,$efficiency_rounded")
            end
        end
    end
    
    @printf("Results saved to %s\n", filename)
end

function main()
    println("=" ^ 80)
    println("N-body Simulation Strong Scaling Benchmark")
    println("=" ^ 80)
    println()
    
    # Strong scaling: fixed problem size, varying number of processes
    # Problem size: 128 bodies, 1000000 iterations, 1 run
    nbodies = 1024
    iterations = 75000
    runs = 1
    
    println("Strong Scaling Configuration:")
    @printf("  Bodies: %d\n", nbodies)
    @printf("  Iterations: %d\n", iterations)
    @printf("  Runs: %d\n", runs)
    println()
    
    # Test only 1, 2, 4, 8 nodes (1 process per node for simplicity)
    # Total processes = number of nodes
    nodes_to_test = [1, 2, 4, 8]
    
    println("Testing nodes: ", join(nodes_to_test, ", "))
    println("  (1 process per node)")
    println()
    println("=" ^ 80)
    println()
    
    results = []  # (nbodies, iterations, nprocs, time)
    
    # Run sequential first (for baseline)
    println("Running sequential baseline...")
    seq_time = benchmark_sequential(nbodies, iterations, runs)
    if seq_time !== nothing
        push!(results, (nbodies, iterations, 0, seq_time))
    end
    println()
    
    # Run parallel versions for each node count
    for nodes in nodes_to_test
        nprocs = nodes  # 1 process per node
        @printf("Testing %d node(s) (%d process(es))...\n", nodes, nprocs)
        
        par_time = benchmark_config(nbodies, iterations, nprocs, runs)
        if par_time !== nothing
            push!(results, (nbodies, iterations, nprocs, par_time))
        end
        println()
    end
    
    # Print results table
    print_results_table(results)
    
    # Print efficiency analysis
    print_efficiency_analysis(results)
    
    # Save results to CSV
    save_results_to_csv(results)
    
    println("Strong scaling benchmark complete!")
end

main()

