import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
from geneticalgorithm2 import geneticalgorithm2 as ga
from numba import njit

INCH_TO_M = 0.0254

CONFIG = (
    0.059, 0.059, 0.04, 0.068, 0.068, 0.013, 0.227, 9.81,
    14.5 * INCH_TO_M, 3.0 * INCH_TO_M, 5.5 * INCH_TO_M,
    16.5 * INCH_TO_M, 3.0 * INCH_TO_M,
    -20.0, 60.0, 40
)
BAND_K0 = 120.0
BAND_CIRCUMFERENCE = 7.0 * INCH_TO_M
MAX_STRETCH = 6.0 * INCH_TO_M
L_MAX = BAND_CIRCUMFERENCE + MAX_STRETCH
theta_vals = np.linspace(np.radians(CONFIG[13]), np.radians(CONFIG[14]), CONFIG[15])

VALID_HOLES_RANGE = {
    (0,0): (0, 27),
    (1,0): (0, 29),
    (0,1): (0, 31),
    (1,1): (2, 31)
}

@njit
def get_pivots(theta):
    c, s = np.cos(theta), np.sin(theta)
    W_l = CONFIG[9]
    L_l = CONFIG[8]
    W_up = CONFIG[10]
    W_u = CONFIG[12]
    L_u = CONFIG[11]
    P0 = np.array([0.0, 0.0])
    P1 = np.array([0.0, W_l])
    vl = np.array([L_l * c, L_l * s])
    P2 = P0 + vl
    P3 = P1 + vl
    P4 = P2 + np.array([0.0, W_up])
    P5 = P4 + np.array([0.0, W_u])
    vu = np.array([-L_u * c, L_u * s])
    P6 = P4 + vu
    P7 = P5 + vu
    return P0, P1, P2, P3, P4, P5, P6, P7

@njit
def get_point(link_id, hole, theta):
    P0, P1, P2, P3, P4, P5, P6, P7 = get_pivots(theta)
    if link_id == 0:
        return P0 + (hole / 29.0) * (P2 - P0)
    elif link_id == 1:
        return P1 + (hole / 29.0) * (P3 - P1)
    elif link_id == 3:
        return P4 + (hole / 33.0) * (P6 - P4)
    elif link_id == 4:
        return P5 + (hole / 33.0) * (P7 - P5)
    else:
        return np.array([0.0, 0.0])

@njit
def gravity_moment(theta):
    P0, _, _, _, _, _, _, _ = get_pivots(theta)
    g = CONFIG[7]
    m = np.array([CONFIG[0], CONFIG[1], CONFIG[2], CONFIG[3], CONFIG[4], CONFIG[5], CONFIG[6]])
    lids = np.array([0,1,2,3,4,5,5])
    holes = np.array([14.5, 14.5, 10.0, 16.5, 16.5, 3.5, 3.5])
    M = 0.0
    for i in range(len(m)):
        pos = get_point(lids[i], holes[i], theta)
        F = np.array([0.0, -m[i] * g])
        r = pos - P0
        M += r[0]*F[1] - r[1]*F[0]
    return M

@njit
def triangle_perimeter(lA, hA, lB, hB, lC, hC, theta):
    pA = get_point(lA, hA, theta)
    pB = get_point(lB, hB, theta)
    pC = get_point(lC, hC, theta)
    dAB = np.sqrt((pB[0]-pA[0])**2 + (pB[1]-pA[1])**2)
    dBC = np.sqrt((pC[0]-pB[0])**2 + (pC[1]-pB[1])**2)
    dCA = np.sqrt((pA[0]-pC[0])**2 + (pA[1]-pC[1])**2)
    return dAB + dBC + dCA

@njit
def triangle_max_perimeter(lA, hA, lB, hB, lC, hC):
    max_p = 0.0
    for th in theta_vals:
        p = triangle_perimeter(lA, hA, lB, hB, lC, hC, th)
        if p > max_p:
            max_p = p
    return max_p

@njit
def triangle_potential_energy(lA, hA, lB, hB, lC, hC, k, L0, n_angles):
    U = np.zeros(n_angles)
    for i in range(n_angles):
        th = theta_vals[i]
        perim = triangle_perimeter(lA, hA, lB, hB, lC, hC, th)
        stretch = max(perim - L0, 0.0)
        U[i] = 0.5 * k * stretch * stretch
    return U

@njit
def triangle_moment(lA, hA, lB, hB, lC, hC, k, L0, n_angles):
    U = triangle_potential_energy(lA, hA, lB, hB, lC, hC, k, L0, n_angles)
    M = np.zeros(n_angles)
    for i in range(1, n_angles-1):
        M[i] = -(U[i+1] - U[i-1]) / (theta_vals[i+1] - theta_vals[i-1])
    M[0] = -(U[1] - U[0]) / (theta_vals[1] - theta_vals[0])
    M[-1] = -(U[-1] - U[-2]) / (theta_vals[-1] - theta_vals[-2])
    return M

@njit
def compute_net_moment(triangles_array, n_active):
    n_angles = len(theta_vals)
    M_grav = np.zeros(n_angles)
    for i in range(n_angles):
        M_grav[i] = gravity_moment(theta_vals[i])
    M_tri = np.zeros(n_angles)
    for t in range(n_active):
        lA = int(triangles_array[t, 0])
        hA = int(triangles_array[t, 1])
        lB = int(triangles_array[t, 2])
        hB = int(triangles_array[t, 3])
        lC = int(triangles_array[t, 4])
        hC = int(triangles_array[t, 5])
        k  = triangles_array[t, 6] * BAND_K0
        L0 = BAND_CIRCUMFERENCE
        M_tri += triangle_moment(lA, hA, lB, hB, lC, hC, k, L0, n_angles)
    return M_grav + M_tri

@njit
def online_fitness(triangles_array, n_active):
    net = compute_net_moment(triangles_array, n_active)
    return np.sqrt(np.mean(net**2))

MAX_TRIANGLES = 4

def decode_ga_vector(x):
    tri_list = []
    for i in range(MAX_TRIANGLES):
        start = i * 7
        active = int(x[start])
        if active == 0:
            continue
        par = int(x[start+1])
        h0 = int(x[start+2])
        h1 = int(x[start+3])
        link2 = int(x[start+4])
        h2 = int(x[start+5])
        n = int(x[start+6])
        if par == 0:
            lA, lB = 0, 1
            max0 = VALID_HOLES_RANGE[(0,0)][1]
            max1 = VALID_HOLES_RANGE[(1,0)][1]
            third_link = lA if link2 == 0 else lB
            max_third = max0 if link2 == 0 else max1
            min_third = 0
        else:
            lA, lB = 3, 4
            max0 = VALID_HOLES_RANGE[(0,1)][1]
            max1 = VALID_HOLES_RANGE[(1,1)][1]
            third_link = lA if link2 == 0 else lB
            max_third = max0 if link2 == 0 else max1
            min_third = VALID_HOLES_RANGE[(0,1)][0] if link2 == 0 else VALID_HOLES_RANGE[(1,1)][0]
        h0 = min(max(h0, 0), max0)
        h1 = min(max(h1, 0), max1)
        h2 = min(max(h2, min_third), max_third)
        if par == 1:
            h0 = max(h0, VALID_HOLES_RANGE[(0,1)][0])
            h1 = max(h1, VALID_HOLES_RANGE[(1,1)][0])
        tri_list.append( (lA, h0, lB, h1, third_link, h2, n) )
    return tri_list

def check_validity(tri_list):
    all_points = {}
    for (lA, hA, lB, hB, lC, hC, n) in tri_list:
        for link, hole in [(lA, hA), (lB, hB), (lC, hC)]:
            if link not in all_points:
                all_points[link] = []
            all_points[link].append(hole)
    for link, holes in all_points.items():
        if len(holes) != len(set(holes)):
            return False
        for i in range(len(holes)):
            for j in range(i+1, len(holes)):
                if abs(holes[i] - holes[j]) < 2:
                    return False
    for (lA, hA, lB, hB, lC, hC, n) in tri_list:
        if (lA == lB and hA == hB) or (lA == lC and hA == hC) or (lB == lC and hB == hC):
            return False
        if triangle_max_perimeter(lA, hA, lB, hB, lC, hC) > L_MAX:
            return False
    return True

def ga_fitness(x):
    tri_list = decode_ga_vector(x)
    if tri_list is None or len(tri_list) == 0:
        return 1e6
    if not check_validity(tri_list):
        return 1e6
    tri_array = np.zeros((len(tri_list), 7), dtype=np.float64)
    for i, (lA, hA, lB, hB, lC, hC, n) in enumerate(tri_list):
        tri_array[i] = [lA, hA, lB, hB, lC, hC, n]
    n_active = tri_array.shape[0]
    f = online_fitness(tri_array, n_active)
    return f if f < 1e6 else 1e6

dim = MAX_TRIANGLES * 7
varbound = []
for i in range(MAX_TRIANGLES):
    varbound.extend([
        [0,1],
        [0,1],
        [0,31],
        [0,31],
        [0,1],
        [0,31],
        [1,6]
    ])
varbound = np.array(varbound[:dim])
vartype = np.array(['int'] * dim)

algorithm_param = {
    'max_num_iteration': None,
    'population_size': 1500,
    'mutation_probability': 0.2,
    'elit_ratio': 0.05,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 150
}

bg_rmse = online_fitness(np.empty((0,7)), 0)
print(f"Baseline gravity RMSE: {bg_rmse:.4f} N·m")

model_ga = ga(dimension=dim, variable_type=vartype,
              variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model_ga.run(function=ga_fitness)

best_x = model_ga.output_dict['variable']
best_tri = decode_ga_vector(best_x)
best_array = np.zeros((len(best_tri), 7), dtype=np.float64)
for i, (lA, hA, lB, hB, lC, hC, n) in enumerate(best_tri):
    best_array[i] = [lA, hA, lB, hB, lC, hC, n]
best_rmse = online_fitness(best_array, len(best_tri))

LINK_NAMES = ["L0", "L1", "RC", "L2", "L3", "LC"]
print("\nOPTIMAL TRIANGLE BANDS")
for i, (lA, hA, lB, hB, lC, hC, n) in enumerate(best_tri):
    max_perim = triangle_max_perimeter(lA, hA, lB, hB, lC, hC)
    print(f"Triangle {i+1}: {LINK_NAMES[lA]} hole {hA} – {LINK_NAMES[lB]} hole {hB} – {LINK_NAMES[lC]} hole {hC}, n={n}, k={n*120} N/m, max perimeter={max_perim:.3f} m")
print(f"\nAchieved RMSE = {best_rmse:.4f} N·m  (baseline {bg_rmse:.4f})")
print(f"Reduction: {(bg_rmse - best_rmse)/bg_rmse*100:.1f}%")

theta_deg = np.degrees(theta_vals)
grav_all = np.array([gravity_moment(t) for t in theta_vals])
net_all = compute_net_moment(best_array, len(best_tri))

fig = plt.figure(figsize=(14, 8))
ax_moment = plt.subplot(2, 2, (1, 2))
ax_moment.plot(theta_deg, grav_all, 'k-', linewidth=2, label='Gravity only')
ax_moment.plot(theta_deg, net_all, 'b--', linewidth=2, label='Net moment (optimised)')
ax_moment.axhline(0, color='gray', linestyle=':')
line_theta, = ax_moment.plot([theta_deg[15]]*2, [net_all.min(), net_all.max()], 'r-', linewidth=1)
ax_moment.set_xlabel('θ (degrees)')
ax_moment.set_ylabel('Moment (N·m)')
ax_moment.set_title('Moment balance with triangle bands')
ax_moment.legend()
ax_moment.grid(True)

ax_mech = plt.subplot(2, 2, (3, 4))
ax_mech.set_aspect('equal')
ax_mech.set_xlim(-0.6, 0.6)
ax_mech.set_ylim(-0.1, 1.0)
ax_mech.set_title('Double reverse parallelogram lift')
ax_mech.grid(True, linestyle=':', alpha=0.5)

bar_width = 1.0
def get_bar_corners(p_start, p_end, width_inches):
    w = (width_inches / 2.0) * INCH_TO_M
    vec = p_end - p_start
    norm = np.array([-vec[1], vec[0]])
    length = np.linalg.norm(norm)
    if length > 0:
        norm = norm / length
    return np.array([p_start - w*norm, p_end - w*norm, p_end + w*norm, p_start + w*norm])

init_theta = np.radians(35)
P0, P1, P2, P3, P4, P5, P6, P7 = get_pivots(init_theta)

bars = []
for p_s, p_e, color in [(P0, P2, '#2196F3'), (P1, P3, '#2196F3'), (P2, P5, '#FF9800'),
                         (P4, P6, '#4CAF50'), (P5, P7, '#4CAF50'), (P6, P7, '#FF9800')]:
    poly = Polygon(get_bar_corners(p_s, p_e, bar_width), closed=True, facecolor=color,
                   edgecolor='black', linewidth=1.5, alpha=0.6)
    ax_mech.add_patch(poly)
    bars.append((p_s, p_e, poly))

tri_elements = []
for (lA, hA, lB, hB, lC, hC, n) in best_tri:
    pA = get_point(lA, hA, init_theta)
    pB = get_point(lB, hB, init_theta)
    pC = get_point(lC, hC, init_theta)
    poly = Polygon([pA, pB, pC], closed=True, edgecolor='red', linestyle='--', linewidth=2, facecolor='none')
    ax_mech.add_patch(poly)
    dotA, = ax_mech.plot(pA[0], pA[1], 'ro', markersize=8)
    dotB, = ax_mech.plot(pB[0], pB[1], 'ro', markersize=8)
    dotC, = ax_mech.plot(pC[0], pC[1], 'ro', markersize=8)
    tri_elements.append((poly, dotA, dotB, dotC, lA, hA, lB, hB, lC, hC))

moment_text = ax_mech.text(0.02, 0.95, '', transform=ax_mech.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
theta_slider = Slider(ax_slider, 'θ (deg)', CONFIG[13], CONFIG[14], valinit=35, valstep=1)

def update(val):
    th_deg = theta_slider.val
    th_rad = np.radians(th_deg)
    P0n, P1n, P2n, P3n, P4n, P5n, P6n, P7n = get_pivots(th_rad)
    new_pivots = [(P0n, P2n), (P1n, P3n), (P2n, P5n), (P4n, P6n), (P5n, P7n), (P6n, P7n)]
    for i, (p_s, p_e) in enumerate(new_pivots):
        bars[i][2].set_xy(get_bar_corners(p_s, p_e, bar_width))
    for poly, dotA, dotB, dotC, lA, hA, lB, hB, lC, hC in tri_elements:
        newA = get_point(lA, hA, th_rad)
        newB = get_point(lB, hB, th_rad)
        newC = get_point(lC, hC, th_rad)
        poly.set_xy([newA, newB, newC])
        dotA.set_data([newA[0]], [newA[1]])
        dotB.set_data([newB[0]], [newB[1]])
        dotC.set_data([newC[0]], [newC[1]])
    grav = gravity_moment(th_rad)
    net = compute_net_moment(best_array, len(best_tri))
    idx = np.argmin(np.abs(theta_deg - th_deg))
    moment_text.set_text(f'θ = {th_deg:.0f}°\nGravity moment = {grav:.3f} N·m\nNet moment = {net[idx]:.3f} N·m')
    line_theta.set_xdata([th_deg, th_deg])
    fig.canvas.draw_idle()

theta_slider.on_changed(update)
update(35)

plt.subplots_adjust(bottom=0.1)

all_x = []
all_y = []
for link_id, n_holes in {0:30, 1:30, 3:34, 4:34}.items():
    for hole in range(n_holes):
        pt = get_point(link_id, hole, init_theta)
        all_x.append(pt[0])
        all_y.append(pt[1])
min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)
pad_x = (max_x - min_x) * 0.15
pad_y = (max_y - min_y) * 0.15

fig2, ax2 = plt.subplots(figsize=(14, 12))
ax2.set_aspect('equal')
ax2.set_xlim(min_x - pad_x, max_x + pad_x)
ax2.set_ylim(min_y - pad_y, max_y + pad_y)
ax2.set_title('C-channel holes and rubber band attachment points (θ=35°)', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.4)

for p_s, p_e, color in [(P0, P2, '#2196F3'), (P1, P3, '#2196F3'), (P2, P5, '#FF9800'),
                         (P4, P6, '#4CAF50'), (P5, P7, '#4CAF50'), (P6, P7, '#FF9800')]:
    ax2.add_patch(Polygon(get_bar_corners(p_s, p_e, bar_width), closed=True, facecolor=color,
                          edgecolor='black', linewidth=2.0, alpha=0.4))

link_colors = {0: '#1565C0', 1: '#1565C0', 3: '#2E7D32', 4: '#2E7D32'}
for link_id, n_holes in {0:30, 1:30, 3:34, 4:34}.items():
    for hole in range(n_holes):
        pt = get_point(link_id, hole, init_theta)
        ax2.plot(pt[0], pt[1], 'o', color=link_colors[link_id], markersize=5, alpha=0.8)
        offset_x = 0.008
        offset_y = 0.008
        if link_id in (0, 3):
            offset_x = -0.015
        ax2.text(pt[0] + offset_x, pt[1] + offset_y, str(hole), fontsize=5, color=link_colors[link_id], alpha=0.7)

colors_tri = ['red', 'darkorange', 'magenta', 'cyan']
for idx, (lA, hA, lB, hB, lC, hC, n) in enumerate(best_tri):
    pts = [get_point(lA, hA, init_theta), get_point(lB, hB, init_theta), get_point(lC, hC, init_theta)]
    color = colors_tri[idx % len(colors_tri)]
    ax2.add_patch(Polygon(pts, closed=True, edgecolor=color, linestyle='-', linewidth=2.5, facecolor=color, alpha=0.15))
    for j, (pt, hole) in enumerate(zip(pts, [hA, hB, hC])):
        ax2.plot(pt[0], pt[1], 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        ax2.text(pt[0] + 0.01, pt[1] + 0.01, f'T{idx+1}:{hole}', fontsize=8, color=color, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0', markersize=10, label='Lower links (L0, L1)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32', markersize=10, label='Upper links (L2, L3)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF9800', markersize=10, label='Vertical bars (RC, LC)'),
    plt.Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=10, label='Band attachment points'),
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()