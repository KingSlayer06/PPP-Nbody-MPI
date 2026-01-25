# N-body simulation for PPP practical

using Printf
using Random

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

function compute_forces!(bodies)
    clear_forces!(bodies)

    # Incrementally accumulate forces from each body pair,
    # skipping force of body on itself (c == b)
    @inbounds for b in 1:length(bodies.force)

        #Get data for body b
        mass_b = bodies.mass[b]
        pos_b = bodies.pos[b]
        force_b = bodies.force[b]

        # We have order N*N/2 force calculations
        # with N = length(bodies.force)
        @inbounds for c in b + 1:length(bodies.force)

            #Get data for body c
            pos_c = bodies.pos[c]
            mass_c = bodies.mass[c]
            force_c = bodies.force[c]

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

            # Add contribution to c
            fx = force_c.x - dfx
            fy = force_c.y - dfy
            fz = force_c.z - dfz
            bodies.force[c] = Vec(fx,fy,fz)

            # Update contribution to b
            fx = force_b.x + dfx
            fy = force_b.y + dfy
            fz = force_b.z + dfz
            force_b = Vec(fx,fy,fz)

        end

        # Add contribution to b
        bodies.force[b] = force_b

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

function perf_nbodies(nbodies, iterations, verbose=false, silent=false)
    bodies = alloc_bodies(nbodies)
    init_bodies!(bodies)

    dt = 0.01
    secs = @elapsed begin
        for i in 1:iterations
            if verbose
                print_bodies(bodies, i)
            end
            compute_forces!(bodies)
            compute_positions!(bodies, dt)
            compute_velocities!(bodies, dt)
        end
    end
    
    @printf("nbody took %.3f secs\n", secs)

    if ! silent
        print_bodies(bodies, iterations+1)
    end
end
