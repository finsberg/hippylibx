from pathlib import Path

import dolfinx

try:
    # dolfinx version >= 0.10.0
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio


def create_circle_mesh(outfile: Path, radius: float = 5, mesh_size: float = 0.2):
    import gmsh

    gmsh.initialize()
    gmsh.model.add("circle")

    vol = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)

    # Synchronize the OCC model with the Gmsh model
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    gmsh.model.mesh.generate(2)
    # Dolfinx needs physical groups to identify parts of the mesh
    gmsh.model.addPhysicalGroup(2, [vol], 1)

    output_file = str(outfile)
    gmsh.write(output_file)
    print(f"Mesh saved to {output_file}")
    gmsh.finalize()


def main(comm, outfile=Path(__file__).parent / "meshes" / "circle.xdmf"):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    msh_file = outfile.with_suffix(".msh")
    if comm.rank == 0:
        create_circle_mesh(outfile=msh_file)
    comm.Barrier()
    res = gmshio.read_from_msh(filename=msh_file, comm=comm)

    with dolfinx.io.XDMFFile(comm, outfile, "w") as xdmf:
        if hasattr(res, "mesh"):
            # dolfinx >= 0.10.0 returns a named tuple
            xdmf.write_mesh(res.mesh)
        else:
            xdmf.write_mesh(res[0])


if __name__ == "__main__":
    main()
