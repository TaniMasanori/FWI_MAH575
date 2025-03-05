# CLAUDE.md - Coding Guidelines and Commands

## Build and Run Commands
- SPECFEM2D compilation:
  ```bash
  ./configure FC=gfortran CC=gcc  # Serial build
  ./configure FC=gfortran CC=gcc MPIFC=mpif90 --with-mpi  # Parallel build
  make all  # Build executables in ./bin/
  ```

- SPECFEM2D execution:
  ```bash
  ./bin/xmeshfem2D > OUTPUT_FILES/mesher_log.txt
  ./bin/xspecfem2D > OUTPUT_FILES/solver_log.txt
  ```

- Seisflows:
  ```bash
  conda activate seisflows
  seisflows setup -f; seisflows configure
  seisflows submit  # Run workflow
  seisflows restart  # Restart workflow
  ```

## Code Style Guidelines
- **Imports**: System imports first, then third-party packages
- **Naming**: snake_case for functions/variables, UPPER_CASE for constants
- **Documentation**: Use docstrings with parameter descriptions
- **Error handling**: Use assertions for validation, specific error messages
- **Plotting**: Use matplotlib with consistent color schemes
- **Notebooks**: Follow sequential execution pattern with markdown documentation

## Docker/Container Usage
```bash
# Mac Intel, Windows, Linux
docker run -v ${PWD}:/notebooks --rm -p 8888:8888 arianoes/specfem2d_jn:amd64
# Mac M1
docker run -v ${PWD}:/notebooks --rm -p 8888:8888 arianoes/specfem2d_jn:arm64
```