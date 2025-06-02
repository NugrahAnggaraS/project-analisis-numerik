import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csc_matrix, linalg
from scipy.spatial import Delaunay
import time

class WaveSimulator2D:
    """
    Simulator gelombang 2D menggunakan FEM untuk diskretisasi ruang
    dan time-stepping eksplisit (Leapfrog scheme)
    
    Menyelesaikan: ∂²u/∂t² = c²∇²u
    """
    
    def __init__(self, Lx=2.0, Ly=2.0, nx=30, ny=30, c=1.0, dt=0.01):
        """
        Parameter:
        - Lx, Ly: dimensi domain
        - nx, ny: jumlah titik grid
        - c: kecepatan gelombang
        - dt: time step
        """
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.c = c
        self.dt = dt
        
        # Buat mesh triangular
        self.create_mesh()
        
        # Hitung matriks FEM
        self.assemble_matrices()
        
        # Inisialisasi variabel waktu
        self.t = 0.0
        self.time_history = []
        self.displacement_history = []
        
    def create_mesh(self):
        """Membuat mesh triangular untuk FEM"""
        # Grid points
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Flatten untuk triangulasi
        points = np.column_stack([X.flatten(), Y.flatten()])
        
        # Delaunay triangulation
        self.tri = Delaunay(points)
        self.points = points
        self.n_nodes = len(points)
        
        # Identifikasi boundary nodes untuk kondisi batas Dirichlet
        self.boundary_nodes = self.find_boundary_nodes()
        
    def find_boundary_nodes(self):
        """Menemukan node pada boundary"""
        boundary = []
        for i, (x, y) in enumerate(self.points):
            if (abs(x) < 1e-10 or abs(x - self.Lx) < 1e-10 or 
                abs(y) < 1e-10 or abs(y - self.Ly) < 1e-10):
                boundary.append(i)
        return np.array(boundary)
    
    def assemble_matrices(self):
        """Assembling matriks massa dan kekakuan untuk FEM"""
        print("Assembling FEM matrices...")
        
        n = self.n_nodes
        M = np.zeros((n, n))  # Mass matrix
        K = np.zeros((n, n))  # Stiffness matrix
        
        # Loop melalui setiap elemen triangular
        for element in self.tri.simplices:
            # Koordinat titik elemen
            coords = self.points[element]
            
            # Hitung matriks elemen lokal
            Me, Ke = self.element_matrices(coords)
            
            # Assembly ke matriks global
            for i in range(3):
                for j in range(3):
                    M[element[i], element[j]] += Me[i, j]
                    K[element[i], element[j]] += Ke[i, j]
        
        # Convert ke sparse matrix untuk efisiensi
        self.M = csc_matrix(M)
        self.K = csc_matrix(K)
        
        # Inverse mass matrix untuk explicit time stepping
        M_diag = np.array(M.diagonal())
        # Lumped mass matrix (diagonal)
        self.M_inv = 1.0 / M_diag
        
    def element_matrices(self, coords):
        """Hitung matriks massa dan kekakuan elemen triangular"""
        # Koordinat nodes elemen
        x1, y1 = coords[0]
        x2, y2 = coords[1] 
        x3, y3 = coords[2]
        
        # Area elemen
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        # Mass matrix elemen (consistent)
        Me = (area/12.0) * np.array([[2, 1, 1],
                                     [1, 2, 1],
                                     [1, 1, 2]])
        
        # Gradien shape functions
        b1 = y2 - y3
        b2 = y3 - y1  
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        
        # Stiffness matrix elemen
        B = np.array([[b1, b2, b3],
                      [c1, c2, c3]]) / (2*area)
        
        Ke = area * np.dot(B.T, B)
        
        return Me, Ke
    
    def set_initial_conditions(self, u0_func, v0_func=None):
        """Set kondisi awal"""
        # Displacement awal
        self.u_prev = np.zeros(self.n_nodes)
        self.u_curr = np.zeros(self.n_nodes)
        
        for i, (x, y) in enumerate(self.points):
            self.u_curr[i] = u0_func(x, y)
        
        # Velocity awal
        if v0_func is not None:
            v0 = np.zeros(self.n_nodes)
            for i, (x, y) in enumerate(self.points):
                v0[i] = v0_func(x, y)
            # Backward Euler untuk step pertama
            self.u_prev = self.u_curr - self.dt * v0
        else:
            self.u_prev = self.u_curr.copy()
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
        
    def apply_boundary_conditions(self):
        """Apply kondisi batas Dirichlet (u = 0 pada boundary)"""
        self.u_curr[self.boundary_nodes] = 0.0
        self.u_prev[self.boundary_nodes] = 0.0
    
    def time_step(self):
        """Satu langkah time stepping menggunakan Leapfrog scheme"""
        # Leapfrog: u^{n+1} = 2u^n - u^{n-1} + dt²c²M⁻¹Ku^n
        
        # Hitung gaya internal
        f_internal = -self.c**2 * self.K.dot(self.u_curr)
        
        # Time stepping
        u_next = (2.0 * self.u_curr - self.u_prev + 
                 self.dt**2 * self.M_inv * f_internal)
        
        # Apply boundary conditions
        u_next[self.boundary_nodes] = 0.0
        
        # Update
        self.u_prev = self.u_curr.copy()
        self.u_curr = u_next.copy()
        self.t += self.dt
        
        # Store history
        self.time_history.append(self.t)
        self.displacement_history.append(self.u_curr.copy())
    
    def simulate(self, T_final, save_interval=10):
        """Jalankan simulasi hingga waktu T_final"""
        print(f"Running simulation for {T_final:.2f} seconds...")
        
        n_steps = int(T_final / self.dt)
        
        for step in range(n_steps):
            self.time_step()
            
            if step % save_interval == 0:
                progress = (step + 1) / n_steps * 100
                print(f"Progress: {progress:.1f}%")
        
        print("Simulation completed!")
    
    def animate_wave(self, figsize=(12, 5), interval=50, colormap='RdBu'):
        """Buat animasi gelombang"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Setup colormap
        if isinstance(colormap, str):
            cmap = plt.cm.get_cmap(colormap)
        else:
            cmap = colormap
        
        # Rekonstruksi grid untuk plotting
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Find data range untuk colorbar
        all_data = np.concatenate(self.displacement_history)
        vmin, vmax = np.min(all_data), np.max(all_data)
        
        # Plot 1: Heatmap
        im1 = ax1.imshow(np.zeros((self.ny, self.nx)), 
                        extent=[0, self.Lx, 0, self.Ly],
                        vmin=vmin, vmax=vmax, cmap=cmap, 
                        origin='lower', animated=True)
        ax1.set_title('Wave Propagation (2D View)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Plot 2: 3D surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, np.zeros_like(X), 
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               alpha=0.8, linewidth=0, antialiased=True)
        ax2.set_title('Wave Propagation (3D View)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('Displacement')
        ax2.set_zlim(vmin, vmax)
        
        # Colorbar
        cbar = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar.set_label('Displacement')
        
        # Time text
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            if frame < len(self.displacement_history):
                u = self.displacement_history[frame]
                t = self.time_history[frame]
                
                # Interpolasi data ke grid regular
                Z = self.interpolate_to_grid(u, X, Y)
                
                # Update 2D plot
                im1.set_array(Z)
                
                # Update 3D plot
                ax2.clear()
                ax2.plot_surface(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax,
                               alpha=0.8, linewidth=0, antialiased=True)
                ax2.set_title(f'Wave Propagation (3D View) - t={t:.3f}s')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y') 
                ax2.set_zlabel('Displacement')
                ax2.set_zlim(vmin, vmax)
                
                # Update time
                time_text.set_text(f'Time: {t:.3f} s')
            
            return [im1, time_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.displacement_history),
                                     interval=interval, blit=False, repeat=True)
        
        plt.tight_layout()
        return fig, anim
    
    def interpolate_to_grid(self, u, X, Y):
        """Interpolasi solusi FEM ke grid regular untuk plotting"""
        from scipy.interpolate import griddata
        
        # Interpolasi menggunakan triangular interpolation
        Z = griddata(self.points, u, (X, Y), method='linear', fill_value=0)
        return Z
    
    def plot_time_series(self, monitor_points=None):
        """Plot time series pada beberapa titik monitoring"""
        if monitor_points is None:
            # Default monitoring points
            monitor_points = [(self.Lx/4, self.Ly/2), 
                            (self.Lx/2, self.Ly/2),
                            (3*self.Lx/4, self.Ly/2)]
        
        # Find nearest nodes untuk monitoring points
        monitor_indices = []
        for point in monitor_points:
            distances = np.sqrt((self.points[:, 0] - point[0])**2 + 
                              (self.points[:, 1] - point[1])**2)
            monitor_indices.append(np.argmin(distances))
        
        # Plot time series
        plt.figure(figsize=(10, 6))
        for i, idx in enumerate(monitor_indices):
            displacement_series = [u[idx] for u in self.displacement_history]
            plt.plot(self.time_history, displacement_series, 
                    label=f'Point {i+1}: ({monitor_points[i][0]:.1f}, {monitor_points[i][1]:.1f})',
                    linewidth=2)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement')
        plt.title('Displacement Time Series at Monitoring Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Fungsi kondisi awal
def gaussian_pulse(x, y, x0=1.0, y0=1.0, sigma=0.2, amplitude=1.0):
    """Pulsa Gaussian sebagai kondisi awal"""
    return amplitude * np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))

def sinusoidal_wave(x, y, kx=2, ky=1, amplitude=0.5):
    """Gelombang sinusoidal sebagai kondisi awal"""
    return amplitude * np.sin(kx*np.pi*x/2) * np.sin(ky*np.pi*y/2)

def circular_wave(x, y, x0=1.0, y0=1.0, freq=3.0, amplitude=0.8):
    """Gelombang melingkar"""
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    return amplitude * np.sin(freq*np.pi*r) * np.exp(-r)

# Demo simulasi
if __name__ == "__main__":
    print("=== Simulasi Perambatan Gelombang 2D dengan FEM ===")
    print("Persamaan: ∂²u/∂t² = c²∇²u")
    print("Metode: FEM + Leapfrog Time Stepping")
    print("Kondisi Batas: Dirichlet (u=0 pada boundary)\n")
    
    # Inisialisasi simulator
    simulator = WaveSimulator2D(
        Lx=2.0, Ly=2.0,      # Domain size
        nx=25, ny=25,         # Grid resolution
        c=2.0,                # Wave speed
        dt=0.005              # Time step
    )
    
    # Set kondisi awal - pilih salah satu:
    print("Setting initial conditions: Gaussian pulse...")
    simulator.set_initial_conditions(
        lambda x, y: gaussian_pulse(x, y, x0=0.5, y0=0.5, sigma=0.15, amplitude=1.0)
    )
    
    # Alternatively, uncomment untuk kondisi awal lain:
    # simulator.set_initial_conditions(
    #     lambda x, y: sinusoidal_wave(x, y, kx=1, ky=1, amplitude=0.8)
    # )
    
    # Jalankan simulasi
    T_final = 2.0  # Total simulation time
    simulator.simulate(T_final, save_interval=5)
    
    # Buat animasi
    print("Creating animation...")
    fig, anim = simulator.animate_wave(figsize=(14, 6), interval=100)
    
    # Plot time series
    simulator.plot_time_series()
    
    # Tampilkan animasi
    plt.show()
    
    print("\n=== Simulasi Selesai ===")
    print(f"Total time steps: {len(simulator.time_history)}")
    print(f"Final time: {simulator.t:.3f} seconds")
    print(f"Number of nodes: {simulator.n_nodes}")
    print(f"Number of elements: {len(simulator.tri.simplices)}")