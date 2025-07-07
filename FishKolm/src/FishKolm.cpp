#include "FishKolm.hpp"

void 
FISHKOLM::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh from " << mesh_file_name << std::endl;

    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }

    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    lhs_matrix.reinit(sparsity);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution;
  }
}

void 
FISHKOLM::assemble_system()
{
  // std::cout << "===============================================" << std::endl;
  // std::cout << "  Assembling the linear system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    update_values | update_gradients | update_quadrature_points | update_normal_vectors |
    update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  lhs_matrix = 0.0;
  system_rhs = 0.0;

  std::vector<double> solution_loc(n_q);
  std::vector<Tensor<1, dim>> solution_gradient_loc(n_q);
  std::vector<double> solution_old_loc(n_q);

  forcing_term.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_rhs = 0.0;

    fe_values.get_function_values(solution, solution_loc);
    fe_values.get_function_gradients(solution, solution_gradient_loc);
    fe_values.get_function_values(solution_old, solution_old_loc);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      const auto &x_q = fe_values.quadrature_point(q);
      const double JxW = fe_values.JxW(q);

      const double alpha = alpha_coefficient.value(x_q);
      const Tensor<2, dim> D_matrix = D.matrix_value(x_q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const double phi_j = fe_values.shape_value(j, q);
          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

          cell_matrix(i, j) += (
            phi_i * phi_j / deltat
            - alpha * phi_j * phi_i
            + 2.0 * alpha * solution_loc[q] * phi_j * phi_i
            + scalar_product(D_matrix * grad_phi_j, grad_phi_i)
          ) * JxW;
        }

        cell_rhs(i) += (
          - (solution_loc[q] - solution_old_loc[q]) / deltat * phi_i
          - scalar_product(D_matrix * solution_gradient_loc[q], grad_phi_i)
          + alpha * solution_loc[q] * (1.0 - solution_loc[q]) * phi_i
        ) * JxW;
      }
    }

    cell->get_dof_indices(dof_indices);

    lhs_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }

  lhs_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

void 
FISHKOLM::solve_linear_system()
{
  SolverControl solver_control(100000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, delta_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

// If having problems with the CG solver, try this alternative method.
void 
FISHKOLM::solve_linear_system_2()
{
  SolverControl solver_control(5000, 1e-6 * system_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  preconditioner.initialize(lhs_matrix, amg_data);

  solver.solve(lhs_matrix, delta_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

unsigned int 
FISHKOLM::solve_newton_opt()
{
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-3;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
  {
    assemble_system();
    residual_norm = system_rhs.l2_norm();

    pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    // We actually solve the system only if the residual is larger than the
    // tolerance.
    if (residual_norm > residual_tolerance)
    {
      solve_linear_system();
      solution_owned += delta_owned;
      solution = solution_owned;
    }
    else
    {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }
  return n_iter;
}

void 
FISHKOLM::solve_newton()
{
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-3;

  unsigned int n_iter = 0;
  double residual_norm = std::numeric_limits<double>::max();

  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = system_rhs.l2_norm();

    pcout << "  Iteration: " << n_iter << ": "
          << " ||r|| = " << std::scientific << std::setprecision(3)
          << residual_norm << std::flush;
    
    if (residual_norm <= residual_tolerance)
    {
      pcout << " < tolerance" << std::endl;
      break;
    }
    
    pcout << std::endl;
    solve_linear_system();
    solution_owned += delta_owned;
    solution = solution_owned;

    ++n_iter;
  }

  if (n_iter == n_max_iters && residual_norm > residual_tolerance)
  {
    pcout << "  Warning: Newton solver did not converge after "
          << n_max_iters << " iterations. Final residual: "
          << residual_norm << std::endl;
  }
}
void 
FISHKOLM::solve_opt()
{
  Timer timer;
  timer.start();
  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T)
  {
    solution_old = solution;

    const unsigned int n_newton_iters = solve_newton_opt();

    if (n_newton_iters >= 1000)
    {
      pcout << "  deny cur step length, reduce length..." << std::endl;
      solution = solution_old; 
      deltat = std::max(deltat * decrease_factor, deltat_min); 
    }
    else 
    {      
      time += deltat;
      ++time_step;
      output(time_step);
      pcout << "  Step " << std::setw(4) << time_step
            << " accepted. Time: " << std::setw(6) << std::fixed << std::setprecision(2) << time
            << " / " << std::fixed << std::setprecision(2) << T
            << " years. (current dt = " << std::scientific << std::setprecision(2) << deltat << ")"
            << std::endl;

      // 4. adjust step length
      if (n_newton_iters < target_newton_iters) {        // convergence too fast,increase length
        deltat = std::min(deltat * increase_factor, deltat_max);
      }
      else if (n_newton_iters > target_newton_iters) {   //convergnece too slow
        deltat = std::max(deltat * decrease_factor, deltat_min);
      }
    }
    pcout <<std::endl;
  }
  
    timer.stop();
    pcout << "total execute time: "<<timer.wall_time()<<" s."<<std::endl;
}
void 
FISHKOLM::solve()
{
  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
  {
    time += deltat;
    ++time_step;

    solution_old = solution;

    pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
          << time << ":" << std::endl;

    solve_newton();
    output(time_step);
  }
}

void 
FISHKOLM::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}

