echo "10790:"
./cuSolverDn_LinearSolver -R='lu' file=10790_A.mtx
echo "=========================================="
echo " "

echo "SC:"
./cuSolverDn_LinearSolver -R='lu' file=SC_A.mtx
echo "=========================================="
echo " "

echo "118:"
./cuSolverDn_LinearSolver -R='lu' file=118_A.mtx
echo "=========================================="
echo " "

echo "14:"
./cuSolverDn_LinearSolver -R='lu' file=14_A.mtx
echo "=========================================="

echo "FJ:"
./cuSolverDn_LinearSolver -R='lu' file=FJ_A.mtx
echo "=========================================="
