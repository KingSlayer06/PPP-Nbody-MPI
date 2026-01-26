# N-body simulation for PPP practical

using Printf
using Random
using MPI

# Vector of three components
struct Vec
    x::Float64
    y::Float64
    z::Float64
end

# Data for ALL bodies
struct Bodies
    pos::Vector{Vec}
    force::Vector{Vec}
    vel::Vector{Vec}
    mass::Vector{Float64}
end

function clear_forces!(bodies)
    z = 0.0
    force = Vec(z,z,z)
    @inbounds for i in 1:length(bodies.force)
        bodies.force[i] = force
    end
end

function get_balanced_body_range(nbodies::Int, rank::Int, nranks::Int)
    # Total number of force pairs
    total_pairs = (nbodies * (nbodies - 1)) ÷ 2
    
    # Pairs per rank
    pairs_per_rank = total_pairs / nranks
    
    # Find which bodies this rank should process
    # Body b computes (nbodies - b) force pairs
    
    accumulated_pairs = 0
    start_body = 0
    end_body = 0
    
    target_start = rank * pairs_per_rank
    target_end = (rank + 1) * pairs_per_rank
    
    for b in 1:nbodies-1
        pairs_for_body = nbodies - b
        
        # Check if this body's pairs overlap with our target range
        if start_body == 0 && accumulated_pairs + pairs_for_body > target_start
            start_body = b
        end
        
        if accumulated_pairs + pairs_for_body >= target_end
            end_body = b
            break
        end
        
        accumulated_pairs += pairs_for_body
    end
    
    if start_body == 0
        start_body = 1
    end
    if end_body == 0 || rank == nranks - 1
        end_body = nbodies - 1
    end
    
    return start_body:end_body
end

function compute_forces_mpi!(bodies, comm, rank, nranks)
    clear_forces!(bodies)

    nbodies = length(bodies.pos)
    local_forces = [Vec(0.0, 0.0, 0.0) for _ in 1:nbodies]

    # Only iterate over bodies assigned to this rank
    @inbounds for b in (rank + 1):nranks:nbodies
        
        # Get data for body b
        mass_b = bodies.mass[b]
        pos_b = bodies.pos[b]
        force_b_local = local_forces[b]

        @inbounds for c in b + 1:nbodies
            # Get data for body c
            pos_c = bodies.pos[c]
            mass_c = bodies.mass[c]
            
            # Distance
            dx = pos_c.x - pos_b.x
            dy = pos_c.y - pos_b.y
            dz = pos_c.z - pos_b.z
            dist_sqr = dx^2 + dy^2 + dz^2
            dist = sqrt(dist_sqr)
            
            # Force between b and c
            # Assume there are no singularities, resulting in division by zero
            # also, we use gravity constant 1 
            coefficient = ((mass_b * mass_c) / (dist * dist_sqr))
            dfx = coefficient * dx
            dfy = coefficient * dy
            dfz = coefficient * dz

            # Accumulate contribution to c
            fx = local_forces[c].x - dfx
            fy = local_forces[c].y - dfy
            fz = local_forces[c].z - dfz
            local_forces[c] = Vec(fx, fy, fz)

            # Accumulate contribution to b
            fx = force_b_local.x + dfx
            fy = force_b_local.y + dfy
            fz = force_b_local.z + dfz
            force_b_local = Vec(fx, fy, fz)
        end

        # Store local contribution for body b
        local_forces[b] = force_b_local
    end

    # Convert to flat arrays for MPI operations
    local_force_array = zeros(Float64, 3 * nbodies)
    global_force_array = zeros(Float64, 3 * nbodies)
    
    for i in 1:nbodies
        local_force_array[3*(i-1) + 1] = local_forces[i].x
        local_force_array[3*(i-1) + 2] = local_forces[i].y
        local_force_array[3*(i-1) + 3] = local_forces[i].z
    end
    
    # Allreduce to sum all partial forces
    MPI.Allreduce!(local_force_array, global_force_array, MPI.SUM, comm)
    
    # Copy results back to bodies structure
    for i in 1:nbodies
        fx = global_force_array[3*(i-1) + 1]
        fy = global_force_array[3*(i-1) + 2]
        fz = global_force_array[3*(i-1) + 3]
        bodies.force[i] = Vec(fx, fy, fz)
    end
end

function compute_forces_mpi_balanced!(bodies, comm, rank, nranks)
    clear_forces!(bodies)

    nbodies = length(bodies.pos)
    local_forces = [Vec(0.0, 0.0, 0.0) for _ in 1:nbodies]

    # Get balanced range of bodies for this rank
    body_range = get_balanced_body_range(nbodies, rank, nranks)

    # Only iterate over bodies assigned to this rank
    @inbounds for b in body_range
        
        # Get data for body b
        mass_b = bodies.mass[b]
        pos_b = bodies.pos[b]
        force_b_local = local_forces[b]

        @inbounds for c in b + 1:nbodies
            # Get data for body c
            pos_c = bodies.pos[c]
            mass_c = bodies.mass[c]
            
            # Distance
            dx = pos_c.x - pos_b.x
            dy = pos_c.y - pos_b.y
            dz = pos_c.z - pos_b.z
            dist_sqr = dx^2 + dy^2 + dz^2
            dist = sqrt(dist_sqr)
            
            # Force between b and c
            # Assume there are no singularities, resulting in division by zero
            # also, we use gravity constant 1 
            coefficient = ((mass_b * mass_c) / (dist * dist_sqr))
            dfx = coefficient * dx
            dfy = coefficient * dy
            dfz = coefficient * dz

            # Accumulate contribution to c
            fx = local_forces[c].x - dfx
            fy = local_forces[c].y - dfy
            fz = local_forces[c].z - dfz
            local_forces[c] = Vec(fx, fy, fz)

            # Accumulate contribution to b
            fx = force_b_local.x + dfx
            fy = force_b_local.y + dfy
            fz = force_b_local.z + dfz
            force_b_local = Vec(fx, fy, fz)
        end

        # Store local contribution for body b
        local_forces[b] = force_b_local
    end

    # Convert to flat arrays for MPI operations
    local_force_array = zeros(Float64, 3 * nbodies)
    global_force_array = zeros(Float64, 3 * nbodies)
    
    for i in 1:nbodies
        local_force_array[3*(i-1) + 1] = local_forces[i].x
        local_force_array[3*(i-1) + 2] = local_forces[i].y
        local_force_array[3*(i-1) + 3] = local_forces[i].z
    end
    
    # Allreduce to sum all partial forces from all ranks
    MPI.Allreduce!(local_force_array, global_force_array, MPI.SUM, comm)
    
    # Copy results back to bodies structure
    for i in 1:nbodies
        fx = global_force_array[3*(i-1) + 1]
        fy = global_force_array[3*(i-1) + 2]
        fz = global_force_array[3*(i-1) + 3]
        bodies.force[i] = Vec(fx, fy, fz)
    end
end

function compute_positions_gather!(bodies, dt, comm, rank, nranks)
    nbodies = length(bodies.pos)
    
    # Each rank computes positions for its subset (cyclic distribution)
    local_count = (nbodies - rank + nranks - 1) ÷ nranks  # Number of bodies this rank handles
    sendbuf = zeros(Float64, local_count * 3)
    
    # Compute and pack local positions
    idx = 1
    @inbounds for b in (rank + 1):nranks:nbodies
        pos_b = bodies.pos[b]
        vel_b = bodies.vel[b]
        sendbuf[3*(idx-1) + 1] = pos_b.x + dt * vel_b.x
        sendbuf[3*(idx-1) + 2] = pos_b.y + dt * vel_b.y
        sendbuf[3*(idx-1) + 3] = pos_b.z + dt * vel_b.z
        idx += 1
    end
    
    # All ranks compute the same counts (cyclic distribution)
    # Each rank sends local_count * 3 floats
    recvbuf = zeros(Float64, nbodies * 3)
    
    # Use Allgather - all ranks send same size, so we can use regular Allgather
    # But counts may vary slightly, so we need Allgatherv with known counts
    sendcounts = Vector{Int32}(undef, nranks)
    for r in 0:nranks-1
        sendcounts[r+1] = Int32(((nbodies - r + nranks - 1) ÷ nranks) * 3)
    end
    recvcounts = sendcounts
    displs = Vector{Int32}(undef, nranks)
    displs[1] = Int32(0)
    for i in 2:nranks
        displs[i] = displs[i-1] + recvcounts[i-1]
    end
    
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recvbuf, recvcounts, displs), comm)
    
    # Unpack directly using cyclic distribution pattern (no need to gather indices)
    @inbounds for r in 0:nranks-1
        offset = displs[r+1]
        local_count_r = sendcounts[r+1] ÷ 3
        idx = 1
        for b in (r + 1):nranks:nbodies
            bodies.pos[b] = Vec(recvbuf[offset + 3*(idx-1) + 1],
                               recvbuf[offset + 3*(idx-1) + 2],
                               recvbuf[offset + 3*(idx-1) + 3])
            idx += 1
        end
    end
end

function compute_velocities_gather!(bodies, dt, comm, rank, nranks)
    nbodies = length(bodies.vel)
    
    # Each rank computes velocities for its subset (cyclic distribution)
    local_count = (nbodies - rank + nranks - 1) ÷ nranks  # Number of bodies this rank handles
    sendbuf = zeros(Float64, local_count * 3)
    
    # Compute and pack local velocities
    idx = 1
    @inbounds for b in (rank + 1):nranks:nbodies
        mass_b = bodies.mass[b]
        force_b = bodies.force[b]
        vel_b = bodies.vel[b]
        dt_div_mass_b = (dt / mass_b)
        sendbuf[3*(idx-1) + 1] = vel_b.x + dt_div_mass_b * force_b.x
        sendbuf[3*(idx-1) + 2] = vel_b.y + dt_div_mass_b * force_b.y
        sendbuf[3*(idx-1) + 3] = vel_b.z + dt_div_mass_b * force_b.z
        idx += 1
    end
    
    # All ranks compute the same counts (cyclic distribution)
    # Each rank sends local_count * 3 floats
    recvbuf = zeros(Float64, nbodies * 3)
    
    # Use Allgather - all ranks send same size, so we can use regular Allgather
    # But counts may vary slightly, so we need Allgatherv with known counts
    sendcounts = Vector{Int32}(undef, nranks)
    for r in 0:nranks-1
        sendcounts[r+1] = Int32(((nbodies - r + nranks - 1) ÷ nranks) * 3)
    end
    recvcounts = sendcounts
    displs = Vector{Int32}(undef, nranks)
    displs[1] = Int32(0)
    for i in 2:nranks
        displs[i] = displs[i-1] + recvcounts[i-1]
    end
    
    MPI.Allgatherv!(sendbuf, MPI.VBuffer(recvbuf, recvcounts, displs), comm)
    
    # Unpack directly using cyclic distribution pattern (no need to gather indices)
    @inbounds for r in 0:nranks-1
        offset = displs[r+1]
        local_count_r = sendcounts[r+1] ÷ 3
        idx = 1
        for b in (r + 1):nranks:nbodies
            bodies.vel[b] = Vec(recvbuf[offset + 3*(idx-1) + 1],
                               recvbuf[offset + 3*(idx-1) + 2],
                               recvbuf[offset + 3*(idx-1) + 3])
            idx += 1
        end
    end
end

function compute_positions!(bodies, dt)
    @inbounds for b in 1:length(bodies.pos)
        pos_b = bodies.pos[b]
        vel_b = bodies.vel[b]
        x = pos_b.x + dt * vel_b.x
        y = pos_b.y + dt * vel_b.y
        z = pos_b.z + dt * vel_b.z
        bodies.pos[b] = Vec(x,y,z)
    end
end

function compute_velocities!(bodies, dt)
    @inbounds for b in 1:length(bodies.vel)
        mass_b = bodies.mass[b]
        force_b = bodies.force[b]
        vel_b = bodies.vel[b]
        dt_div_mass_b = (dt / mass_b)
        vx = vel_b.x + dt_div_mass_b * force_b.x
        vy = vel_b.y + dt_div_mass_b * force_b.y
        vz = vel_b.z + dt_div_mass_b * force_b.z
        bodies.vel[b] = Vec(vx,vy,vz)
    end
end

function print_bodies(bodies, iter)
    @printf("iteration %d\n", iter)
    @printf("     %16s %16s %16s %16s %16s %16s %16s %16s %16s\n",
            "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz")
    for i in 1:length(bodies.pos)
        # @printf("%4d %16.12f %16.12f %16.12f %16.12f %16.12f %16.12f\n",
        @printf("%4d %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n",
                i, bodies.pos[i].x, bodies.pos[i].y, bodies.pos[i].z,
                bodies.vel[i].x, bodies.vel[i].y, bodies.vel[i].z,
                bodies.force[i].x, bodies.force[i].y, bodies.force[i].z)
    end
end

function alloc_bodies(nbodies::Int=5)
    # Allocate bodies
    pos = Vector{Vec}(undef,nbodies)
    force = Vector{Vec}(undef,nbodies)
    vel = Vector{Vec}(undef,nbodies)
    mass = Vector{Float64}(undef,nbodies)
    bodies = Bodies(pos,force,vel,mass)
    return bodies
end

function init_bodies!(bodies)
    # Fill bodies
    nbodies = length(bodies.pos)
    seed = 27102015
    rng = Xoshiro(seed)
    maxcoord = 10.0

    for b in 1:nbodies
        # Set position
        x = randn(rng, Float64) * maxcoord
        y = randn(rng, Float64) * maxcoord
        z = randn(rng, Float64) * maxcoord
        bodies.pos[b] = Vec(x,y,z)

        # Set mass
        radius = 1.0 + ((b * b + 1.0) *
             sqrt((maxcoord * maxcoord) + (maxcoord * maxcoord) + (maxcoord * maxcoord))) /
                  (25.0 * (nbodies * nbodies + 1.0))
        bodies.mass[b] = radius * radius * radius

        # Set velocity
        vx = ((randn(rng, Float64) * 2000.0) - 1000.0) / 2000.0
        vy = ((randn(rng, Float64) * 2000.0) - 1000.0) / 2000.0
        vz = ((randn(rng, Float64) * 2000.0) - 1000.0) / 2000.0
        bodies.vel[b] = Vec(vx,vy,vz)

        # Set force
        fx = 0.0
        fy = 0.0
        fz = 0.0
        bodies.force[b] = Vec(fx,fy,fz)
    end
end

function broadcast_bodies!(bodies, comm, rank)
    nbodies = length(bodies.pos)
    
    # Create flat arrays for broadcasting
    pos_array = zeros(Float64, 3 * nbodies)
    vel_array = zeros(Float64, 3 * nbodies)
    mass_array = zeros(Float64, nbodies)
    
    # Pack data on root
    if rank == 0
        for i in 1:nbodies
            pos_array[3*(i-1) + 1] = bodies.pos[i].x
            pos_array[3*(i-1) + 2] = bodies.pos[i].y
            pos_array[3*(i-1) + 3] = bodies.pos[i].z
            
            vel_array[3*(i-1) + 1] = bodies.vel[i].x
            vel_array[3*(i-1) + 2] = bodies.vel[i].y
            vel_array[3*(i-1) + 3] = bodies.vel[i].z
            
            mass_array[i] = bodies.mass[i]
        end
    end
    
    # Broadcast
    MPI.Bcast!(pos_array, 0, comm)
    MPI.Bcast!(vel_array, 0, comm)
    MPI.Bcast!(mass_array, 0, comm)
    
    # Unpack data on all ranks
    for i in 1:nbodies
        x = pos_array[3*(i-1) + 1]
        y = pos_array[3*(i-1) + 2]
        z = pos_array[3*(i-1) + 3]
        bodies.pos[i] = Vec(x,y,z)
        
        vx = vel_array[3*(i-1) + 1]
        vy = vel_array[3*(i-1) + 2]
        vz = vel_array[3*(i-1) + 3]
        bodies.vel[i] = Vec(vx, vy, vz)

        bodies.mass[i] = mass_array[i]
    end
end

function perf_nbodies(nbodies, iterations, comm, verbose=false, silent=false, balance=false, gather=false)
    # Initialize MPI
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Allocate bodies on all ranks
    bodies = alloc_bodies(nbodies)
    
    # Initialize only on rank 0
    if rank == 0
        init_bodies!(bodies)
    end
    
    # Broadcast initial state to all ranks
    broadcast_bodies!(bodies, comm, rank)
    
    # Synchronize before timing
    MPI.Barrier(comm)
    
    dt = 0.01
    secs = @elapsed begin
        for i in 1:iterations
            if verbose && rank == 0
                print_bodies(bodies, i)
            end
            
            if balance
                compute_forces_mpi_balanced!(bodies, comm, rank, nranks)
            else
                compute_forces_mpi!(bodies, comm, rank, nranks)
            end
    
            if gather
                compute_positions_gather!(bodies, dt, comm, rank, nranks)
                compute_velocities_gather!(bodies, dt, comm, rank, nranks)
            else
                compute_positions!(bodies, dt)
                compute_velocities!(bodies, dt)
            end
        end
        
        # Synchronize after computation
        MPI.Barrier(comm)
    end
    
    # Only rank 0 prints results
    if rank == 0
        @printf("nbody took %.3f secs\n", secs)
        
        if !silent
            print_bodies(bodies, iterations+1)
        end
    end
end