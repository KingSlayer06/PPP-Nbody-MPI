#!/usr/bin/env julia
# Testing script for N-body simulation correctness
# Compares sequential and parallel outputs
#
# Usage:
#   julia testing.jl
#   or
#   ./testing.jl
# run 'export PATH="$HOME/.julia/bin:$PATH"' if mpiexecjl is not found

using Printf

function run_sequential(nbodies, iterations)
    # Capture output by redirecting stdout
    # Note: Don't use -s flag, we need body data for comparison
    output_file = "seq_test_output.txt"
    cmd = `julia -O3 nbody-seq-run.jl $nbodies $iterations 1`
    run(pipeline(cmd, stdout=output_file, stderr=devnull))
    
    # Read and return output
    if isfile(output_file)
        output = read(output_file, String)
        rm(output_file)
        return output
    else
        return ""
    end
end

function run_parallel(nbodies, iterations, nprocs)
    output_file = "par_test_output.txt"
    # Use mpiexecjl if available, otherwise try mpirun
    # Note: Don't use -s flag, we need body data for comparison
    
    # Try mpiexecjl first
    mpiexecjl_cmd = `mpiexecjl -np $nprocs julia -O3 nbody-par-run.jl -balance $nbodies $iterations 1`
    success = false
    try
        run(pipeline(mpiexecjl_cmd, stdout=output_file, stderr=devnull))
        success = true
    catch
        # Fallback to mpirun if mpiexecjl fails
        mpirun_cmd = `mpirun -np $nprocs julia -O3 nbody-par-run.jl $nbodies $iterations 1`
        try
            run(pipeline(mpirun_cmd, stdout=output_file, stderr=devnull))
            success = true
        catch
            success = false
        end
    end
    
    if !success
        return ""
    end
    
    if isfile(output_file)
        output = read(output_file, String)
        rm(output_file)
        return output
    else
        return ""
    end
end

function parse_output(output)
    lines = split(output, '\n')
    bodies_data = []
    time_line = ""
    in_body_section = false
    
    for line in lines
        stripped = strip(line)
        if isempty(stripped)
            continue
        end
        
        if startswith(stripped, "nbody took")
            time_line = stripped
        elseif startswith(stripped, "iteration")
            in_body_section = true
            continue
        elseif in_body_section
            # Skip header line with column names
            if startswith(stripped, "x") || startswith(stripped, "     x")
                continue
            end
            
            # Body data line - format: "   1   x   y   z   vx   vy   vz   fx   fy   fz"
            # Check if line starts with a number (body index)
            parts = split(stripped)
            if length(parts) >= 2
                body_idx = tryparse(Int, parts[1])
                if body_idx !== nothing
                    push!(bodies_data, stripped)
                end
            end
        end
    end
    
    return time_line, bodies_data
end

function compare_outputs(seq_output, par_output, tolerance=1e-6)
    seq_time, seq_bodies = parse_output(seq_output)
    par_time, par_bodies = parse_output(par_output)
    
    differences = []
    
    # Compare number of bodies
    if length(seq_bodies) != length(par_bodies)
        push!(differences, "Different number of bodies: seq=$(length(seq_bodies)), par=$(length(par_bodies))")
        return differences
    end
    
    if isempty(seq_bodies)
        push!(differences, "No body data found in outputs")
        return differences
    end
    
    # Compare each body's data
    for i in 1:length(seq_bodies)
        seq_line = seq_bodies[i]
        par_line = par_bodies[i]
        
        # Extract values (skip body index)
        seq_parts = split(seq_line)
        par_parts = split(par_line)
        
        if length(seq_parts) < 2 || length(par_parts) < 2
            push!(differences, "Body $i: Invalid format")
            continue
        end
        
        # Parse values, skipping body index
        seq_vals = Float64[]
        par_vals = Float64[]
        
        for j in 2:length(seq_parts)
            val = tryparse(Float64, seq_parts[j])
            if val !== nothing
                push!(seq_vals, val)
            end
        end
        
        for j in 2:length(par_parts)
            val = tryparse(Float64, par_parts[j])
            if val !== nothing
                push!(par_vals, val)
            end
        end
        
        if length(seq_vals) != length(par_vals)
            push!(differences, "Body $i: Different number of values (seq=$(length(seq_vals)), par=$(length(par_vals)))")
            continue
        end
        
        # Compare each value
        for j in 1:length(seq_vals)
            diff = abs(seq_vals[j] - par_vals[j])
            if diff > tolerance
                push!(differences, "Body $i, value $j: diff = $diff (seq=$(seq_vals[j]), par=$(par_vals[j]))")
            end
        end
    end
    
    return differences
end

function test_correctness(nbodies, iterations, nprocs=1)
    @printf("Testing with %d bodies, %d iterations, %d process(es)...\n", nbodies, iterations, nprocs)
    
    # Run sequential
    @printf("  Running sequential version...\n")
    seq_output = run_sequential(nbodies, iterations)
    
    if seq_output == ""
        @printf("  ERROR: Sequential run failed!\n")
        return false
    end
    
    # Run parallel
    @printf("  Running parallel version...\n")
    par_output = run_parallel(nbodies, iterations, nprocs)
    
    if par_output == ""
        @printf("  ERROR: Parallel run failed!\n")
        return false
    end
    
    # Compare outputs
    @printf("  Comparing outputs...\n")
    differences = compare_outputs(seq_output, par_output)
    
    if isempty(differences)
        @printf("  ✓ PASS: Outputs match (within tolerance)\n")
        return true
    else
        @printf("  ✗ FAIL: Found %d differences:\n", length(differences))
        for (i, diff) in enumerate(differences)
            if i <= 10  # Show first 10 differences
                @printf("    - %s\n", diff)
            end
        end
        if length(differences) > 10
            @printf("    ... and %d more differences\n", length(differences) - 10)
        end
        return false
    end
end

function main()
    println("=" ^ 60)
    println("N-body Simulation Correctness Testing")
    println("=" ^ 60)
    println()
    
    test_cases = [
        (32, 100, 1),      # Small test
        (32, 1000, 1),     # Medium test
        (100, 100, 1),     # Larger bodies
        (50, 500, 2),      # 2 processes
        (50, 500, 4),      # 4 processes
    ]
    
    passed = 0
    failed = 0
    
    for (nbodies, iterations, nprocs) in test_cases
        println()
        if test_correctness(nbodies, iterations, nprocs)
            passed += 1
        else
            failed += 1
        end
    end
    
    println()
    println("=" ^ 60)
    @printf("Summary: %d passed, %d failed\n", passed, failed)
    println("=" ^ 60)
    
    exit(failed > 0 ? 1 : 0)
end

main()

