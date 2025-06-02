import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings('ignore')

class LaplaceFiniteElementSolver:
    """
    Solver untuk persamaan Laplace menggunakan metode elemen hingga
    pada domain 2D dengan triangulasi
    """
    
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.boundary_nodes = None
        self.solution = None
        
    def generate_l_shape_domain(self, h=0.1):
        """
        Generate L-shaped domain dengan triangulasi
        h: ukuran mesh (semakin kecil semakin detail)
        """
        # Definisi titik-titik L-shape
        # L-shape: persegi panjang besar minus persegi kecil di pojok kanan atas
        
        # Buat grid points untuk L-shape
        x_coords = []
        y_coords = []
        
        # Grid spacing
        nx1, ny1 = int(2/h), int(2/h)  # Grid utama 2x2
        nx2, ny2 = int(1/h), int(1/h)  # Grid yang dipotong 1x1
        
        # Generate points untuk L-shape
        for i in range(nx1 + 1):
            for j in range(ny1 + 1):
                x = i * h
                y = j * h
                
                # Exclude pojok kanan atas (area yang dipotong)
                if not (x >= 1.0 and y >= 1.0):
                    x_coords.append(x)
                    y_coords.append(y)
        
        # Tambahkan beberapa interior points untuk triangulasi yang lebih baik
        interior_points = [
            (0.5, 0.5), (1.5, 0.5), (0.5, 1.5),
            (0.3, 0.3), (0.7, 0.7), (1.3, 0.3),
            (0.25, 0.75), (0.75, 0.25), (1.25, 0.75)
        ]
        
        for x, y in interior_points:
            if not (x >= 1.0 and y >= 1.0):  # Pastikan tidak di area yang dipotong
                x_coords.append(x)
                y_coords.append(y)
        
        points = np.column_stack([x_coords, y_coords])
        
        # Hapus duplikat
        points = np.unique(points, axis=0)
        
        # Delaunay triangulation
        tri_obj = Delaunay(points)
        
        self.nodes = points
        self.elements = tri_obj.simplices
        
        # Identifikasi boundary nodes
        self._identify_boundary_nodes_l_shape()
        
        return self.nodes, self.elements
    
    def generate_circular_domain(self, radius=1.0, h=0.1):
        """
        Generate circular domain dengan triangulasi
        """
        # Generate points dalam lingkaran
        theta = np.linspace(0, 2*np.pi, int(2*np.pi*radius/h), endpoint=False)
        
        # Boundary points
        boundary_x = radius * np.cos(theta)
        boundary_y = radius * np.sin(theta)
        
        # Interior points dengan pola grid
        interior_points = []
        n_interior = int(radius/h)
        
        for i in range(1, n_interior):
            r = i * h
            n_theta = max(6, int(2*np.pi*r/h))
            theta_int = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
            
            for th in theta_int:
                x = r * np.cos(th)
                y = r * np.sin(th)
                if x*x + y*y < radius*radius:
                    interior_points.append([x, y])
        
        # Center point
        interior_points.append([0, 0])
        
        # Combine all points
        all_points = np.vstack([
            np.column_stack([boundary_x, boundary_y]),
            np.array(interior_points)
        ])
        
        # Delaunay triangulation
        tri_obj = Delaunay(all_points)
        
        self.nodes = all_points
        self.elements = tri_obj.simplices
        
        # Boundary nodes (titik pada circumference)
        self.boundary_nodes = []
        for i, (x, y) in enumerate(self.nodes):
            if abs(x*x + y*y - radius*radius) < 1e-10:
                self.boundary_nodes.append(i)
        
        return self.nodes, self.elements
    
    def _identify_boundary_nodes_l_shape(self):
        """
        Identifikasi boundary nodes untuk L-shape domain
        """
        self.boundary_nodes = []
        tolerance = 1e-10
        
        for i, (x, y) in enumerate(self.nodes):
            # Check if point is on boundary of L-shape
            on_boundary = False
            
            # Bottom edge (y = 0, 0 <= x <= 2)
            if abs(y) < tolerance and 0 <= x <= 2:
                on_boundary = True
            
            # Left edge (x = 0, 0 <= y <= 2)  
            elif abs(x) < tolerance and 0 <= y <= 2:
                on_boundary = True
            
            # Top edge (y = 2, 0 <= x <= 1)
            elif abs(y - 2) < tolerance and 0 <= x <= 1:
                on_boundary = True
            
            # Right edge top part (x = 1, 1 <= y <= 2)
            elif abs(x - 1) < tolerance and 1 <= y <= 2:
                on_boundary = True
            
            # Top edge right part (y = 1, 1 <= x <= 2)
            elif abs(y - 1) < tolerance and 1 <= x <= 2:
                on_boundary = True
            
            # Right edge bottom part (x = 2, 0 <= y <= 1)
            elif abs(x - 2) < tolerance and 0 <= y <= 1:
                on_boundary = True
            
            if on_boundary:
                self.boundary_nodes.append(i)
    
    def compute_element_matrices(self, element_nodes):
        """
        Compute element stiffness matrix untuk triangular element
        """
        # Koordinat nodes dari element
        x1, y1 = element_nodes[0]
        x2, y2 = element_nodes[1] 
        x3, y3 = element_nodes[2]
        
        # Area element
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        if area < 1e-12:  # Degenerate element
            return np.zeros((3, 3))
        
        # Gradients dari shape functions
        b1 = y2 - y3
        b2 = y3 - y1  
        b3 = y1 - y2
        
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        
        # Element stiffness matrix
        K_e = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                b_i = [b1, b2, b3][i]
                c_i = [c1, c2, c3][i]
                b_j = [b1, b2, b3][j]
                c_j = [c1, c2, c3][j]
                
                K_e[i, j] = (b_i * b_j + c_i * c_j) / (4 * area)
        
        return K_e
    
    def assemble_global_matrix(self):
        """
        Assemble global stiffness matrix
        """
        n_nodes = len(self.nodes)
        n_elements = len(self.elements)
        
        # Initialize sparse matrix menggunakan lists
        row_idx = []
        col_idx = []
        data = []
        
        for elem_idx, element in enumerate(self.elements):
            # Get element nodes coordinates
            element_nodes = self.nodes[element]
            
            # Compute element matrix
            K_e = self.compute_element_matrices(element_nodes)
            
            # Assembly ke global matrix
            for i in range(3):
                for j in range(3):
                    if abs(K_e[i, j]) > 1e-12:  # Hanya tambahkan non-zero entries
                        row_idx.append(element[i])
                        col_idx.append(element[j])
                        data.append(K_e[i, j])
        
        # Create sparse matrix
        K_global = csr_matrix((data, (row_idx, col_idx)), 
                             shape=(n_nodes, n_nodes))
        
        # Sum duplicate entries
        K_global.sum_duplicates()
        
        return K_global
    
    def apply_boundary_conditions(self, K_global, boundary_values):
        """
        Apply Dirichlet boundary conditions
        boundary_values: dict {node_index: value}
        """
        n_nodes = len(self.nodes)
        f = np.zeros(n_nodes)  # RHS vector (untuk Laplace = 0)
        
        # Convert to dense untuk modifikasi
        K = K_global.toarray()
        
        # Apply boundary conditions menggunakan elimination method
        for node_idx, value in boundary_values.items():
            # Set row
            K[node_idx, :] = 0
            K[node_idx, node_idx] = 1
            f[node_idx] = value
            
            # Modify other rows
            for i in range(n_nodes):
                if i != node_idx:
                    f[i] -= K[i, node_idx] * value
                    K[i, node_idx] = 0
        
        return csr_matrix(K), f
    
    def solve_laplace(self, domain_type='l_shape', boundary_type='dirichlet'):
        """
        Solve persamaan Laplace dengan boundary conditions
        """
        # Generate domain
        if domain_type == 'l_shape':
            self.generate_l_shape_domain(h=0.15)
        elif domain_type == 'circle':
            self.generate_circular_domain(h=0.2)
        
        print(f"Domain generated: {len(self.nodes)} nodes, {len(self.elements)} elements")
        print(f"Boundary nodes: {len(self.boundary_nodes)}")
        
        # Assemble global matrix
        print("Assembling global stiffness matrix...")
        K_global = self.assemble_global_matrix()
        
        # Define boundary conditions
        boundary_values = {}
        
        if boundary_type == 'dirichlet':
            # Dirichlet BC: u = g pada boundary
            for node_idx in self.boundary_nodes:
                x, y = self.nodes[node_idx]
                
                if domain_type == 'l_shape':
                    # Example: temperature distribution
                    if abs(y) < 1e-10:  # Bottom edge
                        boundary_values[node_idx] = 0.0  # Cold
                    elif abs(x) < 1e-10 and y > 1:  # Left edge (upper part)
                        boundary_values[node_idx] = 100.0  # Hot  
                    else:
                        boundary_values[node_idx] = 50.0  # Moderate
                        
                elif domain_type == 'circle':
                    # Sinusoidal boundary condition
                    theta = np.arctan2(y, x)
                    boundary_values[node_idx] = 100 * np.sin(2 * theta)
        
        # Apply boundary conditions
        print("Applying boundary conditions...")
        K_modified, f = self.apply_boundary_conditions(K_global, boundary_values)
        
        # Solve system
        print("Solving linear system...")
        self.solution = spsolve(K_modified, f)
        
        print("Solution completed!")
        return self.solution
    
    def visualize_solution(self, figsize=(15, 5)):
        """
        Visualize solution dengan contour plots
        """
        if self.solution is None:
            print("No solution to visualize. Run solve_laplace() first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Mesh
        axes[0].triplot(self.nodes[:, 0], self.nodes[:, 1], self.elements, 
                       'k-', alpha=0.3, linewidth=0.5)
        axes[0].scatter(self.nodes[self.boundary_nodes, 0], 
                       self.nodes[self.boundary_nodes, 1], 
                       c='red', s=20, label='Boundary nodes')
        axes[0].set_title('Finite Element Mesh')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].legend()
        axes[0].axis('equal')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Contour plot
        triang = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        cs = axes[1].tricontour(triang, self.solution, levels=20, colors='black', 
                               linewidths=0.5, alpha=0.7)
        cf = axes[1].tricontourf(triang, self.solution, levels=20, cmap='RdYlBu_r')
        fig.colorbar(cf, ax=axes[1])
        axes[1].set_title('Solution Contours u(x,y)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].axis('equal')
        
        # Plot 3: 3D surface
        ax_3d = fig.add_subplot(133, projection='3d')
        surf = ax_3d.plot_trisurf(self.nodes[:, 0], self.nodes[:, 1], self.solution,
                                 triangles=self.elements, cmap='RdYlBu_r', alpha=0.8)
        ax_3d.set_title('3D Solution Surface')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y') 
        ax_3d.set_zlabel('u(x,y)')
        fig.colorbar(surf, ax=ax_3d, shrink=0.6)
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        print(f"\nSolution Statistics:")
        print(f"Min value: {np.min(self.solution):.4f}")
        print(f"Max value: {np.max(self.solution):.4f}")
        print(f"Mean value: {np.mean(self.solution):.4f}")
    
    def analyze_convergence(self):
        """
        Analisis konvergensi dengan berbagai ukuran mesh
        """
        mesh_sizes = [0.3, 0.2, 0.15, 0.1]
        n_nodes_list = []
        max_values = []
        
        print("Convergence Analysis:")
        print("-" * 50)
        
        for h in mesh_sizes:
            # Generate mesh dengan ukuran h
            self.generate_l_shape_domain(h=h)
            
            # Solve
            solution = self.solve_laplace(domain_type='l_shape')
            
            n_nodes_list.append(len(self.nodes))
            max_values.append(np.max(solution))
            
            print(f"h = {h:.3f}, Nodes = {len(self.nodes):4d}, Max u = {np.max(solution):.6f}")
        
        # Plot convergence
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.loglog(mesh_sizes, n_nodes_list, 'bo-')
        plt.xlabel('Mesh size h')
        plt.ylabel('Number of nodes')
        plt.title('Mesh refinement')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.semilogx(mesh_sizes, max_values, 'ro-')
        plt.xlabel('Mesh size h')
        plt.ylabel('Max solution value')
        plt.title('Solution convergence')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Demonstrasi penggunaan
if __name__ == "__main__":
    # Inisialisasi solver
    solver = LaplaceFiniteElementSolver()
    
    print("=== FINITE ELEMENT SOLUTION OF LAPLACE EQUATION ===")
    print("Solving ∇²u = 0 on L-shaped domain")
    print("Using linear triangular elements")
    print()
    
    # Solve untuk L-shape domain
    print("1. Solving on L-shaped domain...")
    solution_l = solver.solve_laplace(domain_type='l_shape', boundary_type='dirichlet')
    solver.visualize_solution()
    
    print("\n" + "="*60)
    
    # Solve untuk circular domain  
    print("2. Solving on circular domain...")
    solver_circle = LaplaceFiniteElementSolver()
    solution_c = solver_circle.solve_laplace(domain_type='circle', boundary_type='dirichlet')
    solver_circle.visualize_solution()
    
    print("\n" + "="*60)
    
    # Convergence analysis
    print("3. Convergence analysis...")
    solver_conv = LaplaceFiniteElementSolver()
    solver_conv.analyze_convergence()