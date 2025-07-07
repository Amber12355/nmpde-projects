#ifndef FISHKOLM_3D_HPP
#define FISHKOLM_3D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class FISHKOLM
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Synthetic tissue map and fiber field (placeholders)
  static bool tissue_is_white(const Point<dim> &p)
  {
    // For testing: white matter area
    return p[0] < 50;
  }

  static Tensor<1, dim> fiber_direction(const Point<dim> & p)
  {
    Tensor<1, dim> n;
    // Simple example: all fibers point in x-direction
    n[0] = 1.0; 
    n[1] = 0.0; 
    n[2] = 0.0;
    return n;

    // n[0] = 1.0;
    // if (p[1] < 0.5)
    //   n[1] = 1.0;
    // else
    //   n[2] = 1.0;
    // return n;
  }

  // ⍺ = Alpha coefficient.
  class FunctionAlpha : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      return FISHKOLM::tissue_is_white(p) ? 0.6 : 0.3;
    }
  };

  // The forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // The matrix D.
  class FunctionD
  {
  public:
    Tensor<2, dim> 
    matrix_value(const Point<dim> & p /* , Tensor<2, dim> &values */ ) const
    {
      // Prion-like spreading models
      const double d_ext = 1.5;
      const double d_axn = FISHKOLM::tissue_is_white(p) ? 3.0 : 0.0;
      Tensor<1, dim> n = FISHKOLM::fiber_direction(p);

      Tensor<2, dim> values;
      for (unsigned int i = 0; i < dim; ++i)
        values[i][i] = d_ext;                      // Extracellular diffusion rate
  
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          values[i][j] += d_axn * n[i] * n[j];     // Axonal transport rate
  
      return values;
    }

    double
    value(const Point<dim> & p,
          const unsigned int component_1 = 0,
          const unsigned int component_2 = 1) const
    {
      return matrix_value(p)[component_1][component_2];
    }
  };

  // Initial conditions.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // if (p[0] < 65 && p[0] > 55 && p[1] < 85 && p[1] > 75 && p[2] < 45 && p[2] > 35)
      // {
      //  return 0.9;
      // }

      // Neocortex area - Alzheimer's disease
      if (p[0] > 40 && p[0] < 80 &&  p[1] > 100 && p[1] < 150 && p[2] > 50 &&  p[2] < 110 )
      {
        return 0.9;
      } else if (p[0] > 20 && p[0] < 60 && p[1] > 10 && p[1] < 50 && p[2] > 30 && p[2] < 80 )
      {
        return 0.9;
      } else {
        return 0.0;
      }
      

      return 0;
    }
  };

  // Exact solution, assume it as zero.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] = 0;
      result[1] = 0;
      if (dim == 3)
      {
        result[2] = 0;
      }

      return result;
    }
  };

  // Constructor. 
  FISHKOLM(const std::string &mesh_file_name_,
           const unsigned int &r_,
           const double &T_,
           double &deltat_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
      , pcout(std::cout, mpi_rank == 0)
      , T(T_)
      , mesh_file_name(mesh_file_name_)
      , r(r_)
      , deltat(deltat_)
      , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();
  void
  solve_opt();

protected:
  // Assemble.
  void
  assemble_system();

  // Solve the linear system.
  void
  solve_linear_system();

  void
  solve_linear_system_2();

  // Solve the problem for one time step.
  void
  solve_newton();
  unsigned int
  solve_newton_opt();
  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Alpha coefficient.
  FunctionAlpha alpha_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // Matrix D.
  FunctionD D;

  // Initial conditions.
  FunctionU0 u_0;

  // Exact solution.
  ExactSolution exact_solution;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  double deltat;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Matrix on the left-hand side.
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;

  double       deltat_min = 1.0 / 36.0;      
  double       deltat_max = 1.0;      
  const double increase_factor = 1.2; 
  const double decrease_factor = 0.5; 

  const unsigned int target_newton_iters = 4;
};

#endif
