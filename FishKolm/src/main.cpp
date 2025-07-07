#include <deal.II/base/convergence_table.h>

#include "FishKolm.hpp"

int  
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../mesh/brain-h3.0.msh";
  const unsigned int degree         = 1;

  double T      = 40.0;            // 40 years
  double deltat = 1.0/3.0;         // four months every time step
  
  FISHKOLM problem(mesh_file_name, degree, T, deltat);
  
  problem.setup();
  problem.solve();

  return 0;
}
