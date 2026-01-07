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

            # Accumulate contribution to c (force on c from b)
            force_c = local_forces[c]
            local_forces[c] = Vec(force_c.x - dfx, force_c.y - dfy, force_c.z - dfz)
            
            # Accumulate contribution to b (force on b from c)
            force_b_local = Vec(force_b_local.x + dfx, 
                               force_b_local.y + dfy, 
                               force_b_local.z + dfz)
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
        bodies.force[i] = Vec(global_force_array[3*(i-1) + 1],
                             global_force_array[3*(i-1) + 2],
                             global_force_array[3*(i-1) + 3])
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

function broadcast_bodies!(bodies, comm)
    nbodies = length(bodies.pos)
    
    # Create flat arrays for broadcasting
    pos_array = zeros(Float64, 3 * nbodies)
    vel_array = zeros(Float64, 3 * nbodies)
    mass_array = zeros(Float64, nbodies)
    
    rank = MPI.Comm_rank(comm)
    
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
        bodies.pos[i] = Vec(pos_array[3*(i-1) + 1],
                           pos_array[3*(i-1) + 2],
                           pos_array[3*(i-1) + 3])
        
        bodies.vel[i] = Vec(vel_array[3*(i-1) + 1],
                           vel_array[3*(i-1) + 2],
                           vel_array[3*(i-1) + 3])
        
        bodies.mass[i] = mass_array[i]
    end
end

function perf_nbodies(nbodies, iterations, comm, verbose=false, silent=false)
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
    broadcast_bodies!(bodies, comm)
    
    # Synchronize before timing
    MPI.Barrier(comm)
    
    # Don't include initialization time
    t1 = time()
    
    dt = 0.01
    for i in 1:iterations
        if verbose && rank == 0
            print_bodies(bodies, i)
        end
        
        compute_forces_mpi!(bodies, comm, rank, nranks)
        compute_positions!(bodies, dt)
        compute_velocities!(bodies, dt)
    end
    
    # Synchronize after computation
    MPI.Barrier(comm)
    
    t2 = time()
    secs = t2 - t1
    
    # Only rank 0 prints results
    if rank == 0
        @printf("nbody took %.3f secs\n", secs)
        
        if !silent
            print_bodies(bodies, iterations+1)
        end
    end
end