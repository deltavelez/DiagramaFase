# PROGRAMA DE DIAGRAMAS DE FASE
# Diego Velez, 2024
# *** Este software se ofrece TAL COMO ESTA sin ninguna garantía ***

import numpy as np
import pygame
import os
import sys
from pygame.locals import *

# Factores de conversión angular
D2R = np.pi/180.0;  R2D = 180.0/np.pi

################################################################################################################################################################
# Establecimiento de la pantalla principal con PyGamme
################################################################################################################################################################
screen_width = 1080
screen_height = 1080
pygame.init()
pygame.display.set_caption('DIAGRAMAS DE FASE ver. 08   DIEGO VELEZ')
pygame.font.init()
my_font = pygame.font.SysFont('arial', 30)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.key.set_repeat(1, 50)

class VP2d:
    xp_min: float
    xp_max: float
    yp_min: float
    yp_max: float
    xr_min: float
    xr_max: float
    yr_min: float
    yr_max: float

################################################################################################################################################################
# Funciones para rotar o proyectar
################################################################################################################################################################
def project_2d(vp2d, R):
    P = np.array([ 0.0, 0.0 ])
    P[0]= vp2d.xp_min + (R[0]-vp2d.xr_min)*(vp2d.xp_max-vp2d.xp_min)/(vp2d.xr_max-vp2d.xr_min)
    P[1]= vp2d.yp_max + (R[1]-vp2d.yr_min)*(vp2d.yp_min-vp2d.yp_max)/(vp2d.yr_max-vp2d.yr_min)
    return P

def project_2d_inv(vp2d, P):
    R = np.array([ 0.0, 0.0 ])
    R[0] = vp2d.xr_min + (P[0]-vp2d.xp_min)*(vp2d.xr_max-vp2d.xr_min)/(vp2d.xp_max-vp2d.xp_min)
    R[1] = vp2d.yr_min + (P[1]-vp2d.yp_max)*(vp2d.yr_max-vp2d.yr_min)/(vp2d.yp_min-vp2d.yp_max)
    return R


################################################################################################################################################################
# Functiones de dibujo
################################################################################################################################################################
# Funcion para escribir texto en pantalla
def draw_text(surface, x, y, message, color):
    text_surface = my_font.render(message, False, color)
    surface.blit(text_surface, (x,y))

def draw_vector(surface,P0, P1, length, angle, width, color):
    pygame.draw.line(surface, color, P0, P1, width)
    if length==0  or (P0[0] == P1[0] and P0[1]==P1[1]):
        return;
    t = 1.0 - length/np.sqrt((P0[0]-P1[0])**2 + (P0[1]-P1[1])**2)
    lx = P1[0] - P0[0] - t*(P1[0]-P0[0])
    ly = P1[1] - P0[1] - t*(P1[1]-P0[1])
    c = np.cos(angle)
    s = np.sin(angle)
    lxp = c*lx - s*ly
    lyp = s*lx + c*ly
    pygame.draw.line(surface, color, P1, [P1[0]-lxp, P1[1]-lyp], width)
    lxp =  c*lx + s*ly
    lyp = -s*lx + c*ly
    pygame.draw.line(surface, color, P1, [P1[0]-lxp, P1[1]-lyp], width)    

def draw_direction(surface,P, radius, angle, color):
    p1x = radius*np.cos(angle)
    p1y = radius*np.sin(angle)

    p2x = radius*np.cos(angle+150*D2R)
    p2y = radius*np.sin(angle+150*D2R)

    p3x = radius*np.cos(angle+210*D2R)
    p3y = radius*np.sin(angle+210*D2R)
    
    pygame.draw.polygon(surface, color, [[P[0]+p1x,P[1]+p1y],
                                         [P[0]+p2x,P[1]+p2y],
                                         [P[0]+p3x,P[1]+p3y]])

def plot_line_2d(surface, vp2d, R0, R1, w, color):
    P0 = project_2d(vp2d,R0)
    P1 = project_2d(vp2d,R1)
    pygame.draw.line(surface, color, P0, P1, w)

def plot_grid(surface, vp2d, delta, w, color):
    x = 0
    while x<vp2d.xr_max:
        P0 = project_2d(vp2d,np.array([x,vp2d.yr_min]))
        P1 = project_2d(vp2d,np.array([x,vp2d.yr_max]))
        pygame.draw.line(surface, color, P0, P1, w)
        x = x + delta

    x = 0
    while x>vp2d.xr_min:
        P0 = project_2d(vp2d,np.array([x,vp2d.yr_min]))
        P1 = project_2d(vp2d,np.array([x,vp2d.yr_max]))
        pygame.draw.line(surface, color, P0, P1, w)
        x = x-delta

    y = 0
    while y<vp2d.yr_max:
        P0 = project_2d(vp2d,np.array([vp2d.xr_min,y]))
        P1 = project_2d(vp2d,np.array([vp2d.xr_max,y]))
        pygame.draw.line(surface, color, P0, P1, w)
        y = y + delta

    y = 0
    while y>vp2d.yr_min:
        P0 = project_2d(vp2d,np.array([vp2d.xr_min,y]))
        P1 = project_2d(vp2d,np.array([vp2d.xr_max,y]))
        pygame.draw.line(surface, color, P0, P1, w)
        y = y-delta


def plot_rectangle_2d(surface, vp2d, R0, R1, w, color):
    P0 = project_2d(vp2d,R0)
    P1 = project_2d(vp2d,R1)
    pygame.draw.line(surface, color, P0, [P1[0],P0[1]], w)
    pygame.draw.line(surface, color, [P1[0],P0[1]], P1, w)
    pygame.draw.line(surface, color, P1, [P0[0],P1[1]], w)
    pygame.draw.line(surface, color, [P0[0],P1[1]], P0, w)

def plot_slope(surface, vp2d, R, y, x, color, radius, width):
    P = project_2d(vp2d,R)
    tetha = np.arctan2(y,x)
    dx = radius*np.cos(tetha)
    dy = radius*np.sin(tetha)
    #pygame.draw.line(surface, color, [P[0]+dx, P[1]-dy], [P[0]-dx, P[1]+dy], width)
    draw_vector(surface, [P[0]-dx, P[1]+dy], [P[0]+dx, P[1]-dy], 0.5*radius, 20*D2R, width, color)

def plot_slope2(surface, vp2d, R, y, x, radius, width, s_max):
    P = project_2d(vp2d,R)
    tetha = np.arctan2(y,x)
    dx = radius*np.cos(tetha)
    dy = radius*np.sin(tetha)
    #pygame.draw.line(surface, color, [P[0]+dx, P[1]-dy], [P[0]-dx, P[1]+dy], width)
    norm = np.sqrt(x**2 + y**2)
    s = 255-int(191*norm/s_max)
    if s<0:
        s=0
    if s>255:
        s=255
    
    draw_vector(surface, [P[0]-dx, P[1]+dy], [P[0]+dx, P[1]-dy], 0.5*radius, 20*D2R, width, (0,s,0))
    
def plot_vector_2d(surface, vp2d, R0, R1, length, angle, width, color):
    P0 = project_2d(vp2d,R0)
    P1 = project_2d(vp2d,R1)
    draw_vector(surface, [P0[0], P0[1]], [P1[0], P1[1]],  length, angle, width, color)

def plot_axis(surface,vp2d):
    plot_vector_2d(surface, vp2d, np.array([vp2d.xr_min,0]), np.array([vp2d.xr_max,0]), 10, 30*D2R, 3, "black")
    plot_vector_2d(surface, vp2d, np.array([0,vp2d.yr_min]), np.array([0,vp2d.yr_max]), 10, 30*D2R, 3, "black")
    P = project_2d(vp2d, np.array([vp2d.xr_max,0]))
    draw_text(screen, P[0]-60, P[1], "{:2.2f}".format(vp2d.xr_max), "black")
    P = project_2d(vp2d, np.array([vp2d.xr_min,0]))
    draw_text(screen, P[0]+10, P[1], "{:2.2f}".format(vp2d.xr_min), "black")
    P = project_2d(vp2d, np.array([0,vp2d.yr_min]))
    draw_text(screen, P[0]+10, P[1]-40, "{:2.2f}".format(vp2d.yr_min), "black")
    P = project_2d(vp2d, np.array([0,vp2d.yr_max]))
    draw_text(screen, P[0]+10, P[1], "{:2.2f}".format(vp2d.yr_max), "black")


################################################################################################################################################################
# Definición del sistema coordenado y de los límites del sistema
################################################################################################################################################################
xa, xb, nx = 0.0, 5.0, 20  
ya, yb, ny = 0.0, 5.0, 20   # Define range and step size for the inner loop
dx = (xb - xa)/nx
dy = (yb - ya)/ny

# Main program
screen.fill((255, 255, 255))
# Creacion del Viewport 2D
vp2d = VP2d()
vp2d.xp_min=0
vp2d.xp_max=screen_width
vp2d.yp_min=0
vp2d.yp_max=screen_height
vp2d.xr_min=xa
vp2d.xr_max=xb
vp2d.yr_min=ya
vp2d.yr_max=yb


################################################################################################################################################################
# Función de analísis de estabilidad para cualquier punto arbitrario, mediante estudio de los autovalores
################################################################################################################################################################
def analize_point(P, flag_data):
    A = np.array([[0.0,0.0],[0.0,0.0]])
    # Build the Jacobian
    delta = 1e-4
    A[0][0] = (dx_dt(P[0]+delta, P[1]) - dx_dt(P[0], P[1]))/delta;   A[0][1] = (dx_dt(P[0], P[1]+delta) - dx_dt(P[0], P[1]))/delta
    A[1][0] = (dy_dt(P[0]+delta, P[1]) - dy_dt(P[0], P[1]))/delta;   A[1][1] = (dy_dt(P[0], P[1]+delta) - dy_dt(P[0], P[1]))/delta

    EL, EV = np.linalg.eig(A)
    
    traza = np.trace(A)
    det = np.linalg.det(A)


    TOL = 1e-2

    if np.abs(det)<TOL:
        if traza<0:
            msg = "Línea de puntos fijos (estables)"
        else:
            msg = "Línea de puntos fijos (inestables)"
    elif det<0:
        msg = "Punto silla"
    else:
        # det >0 
        if np.abs(traza**2-4.0*det)<TOL: 
            if traza<0:
                msg = "R0-: sumidero deg. (estable)"
            else:
                msg = "R0+: fuente deg (inestable)"
        elif traza**2-4*det<0:
                if traza<0:
                    msg = "R1: sumidero espiral (estable)"
                else:
                    msg = "R2: fuente espiral (inestable)"
        else:
            if traza<0:
                msg = "R4: nodo sumidero (estable)"
            else:
                msg = "R3: nodo fuente (inestable)"
    if flag_data:
        print("-------------------")
        print("A = ",A)
        print("Autovalores = ",EL)
        print("Traza = ", traza)
        print("Determinante = ", det)
        print("tr^2 - 4*det = ", traza**2-4.0*det)
        print(msg)
        
    return  msg

################################################################################################################################################################
# Definición de la ecuación diferencial
################################################################################################################################################################
def dx_dt(x, y):
    try:
        alpha = 3.0/2.0
        return alpha*x*y/(1+y)-x
    except:
        return 0.0

def dy_dt(x, y):
    try:
        beta = 4.0
        return -x*y/(1+y)-y+beta
    except:
        return 0.0
    

##def dx_dt(x, y):
##    temp = x**4 -2*x*y**3
##    if np.abs(temp)<1e5:
##        return temp
##    return 0.0
##
##def dy_dt(x, y):
##    temp = 2*(x**3)*y-y**4
##
##    if np.abs(temp)<1e9:
##        return temp
##    return 0


##def dx_dt(x, y):
##    temp = 0.1*x-0.2*y+0.35*0
##    if np.abs(temp)<1e5:
##        return temp
##    return 0.0
##
##def dy_dt(x, y):
##    temp = 0.1*x +0.1*y -0.25*0
##
##    if np.abs(temp)<1e9:
##        return temp
##    return 0

##def dx_dt(x, y):
##    temp = x*y 
##    if np.abs(temp)<1e5:
##        return temp
##    return 0.0
##
##def dy_dt(x, y):
##    temp = 1-x**2-y**2
##
##    if np.abs(temp)<1e9:
##        return temp
##    return 0

################################################################################################################################################################
# Método de Runge Kutta-4 básico, en tiempos positivos 
################################################################################################################################################################
def plot_rk4_pos(surface, color, ta, tb, dt, x0, y0, width, dist_arrow):
    t = ta
    if t<0.0:
        t = 0.0

    n_steps = int(np.abs((tb-t)/dt))
    x_prev = x0
    y_prev = y0
    
    P_prev = project_2d(vp2d, np.array([x0,y0]))
        
    dist = 0.0
    x = x0
    y = y0
    
    for i in range (n_steps):
        # RK4 estimates
        k1_x = dx_dt(x, y)
        k1_y = dy_dt(x, y)
        
        k2_x = dx_dt(x + 0.5*k1_x*dt, y + 0.5*k1_y*dt)
        k2_y = dy_dt(x + 0.5*k1_x*dt, y + 0.5*k1_y*dt)
        
        k3_x = dx_dt(x + 0.5*k2_x*dt, y + 0.5*k2_y*dt)
        k3_y = dy_dt(x + 0.5*k2_x*dt, y + 0.5*k2_y*dt)
        
        k4_x = dx_dt(x + k3_x*dt, y + k3_y*dt)
        k4_y = dy_dt(x + k3_x*dt, y + k3_y*dt)

        # Update x1 and x2 using RK4
        x = x + dt*(k1_x + 2*k2_x + 2*k3_x + k4_x)*0.1666666666
        y = y + dt*(k1_y + 2*k2_y + 2*k3_y + k4_y)*0.1666666666
        P = project_2d(vp2d, np.array([x,y]))
        pygame.draw.line(surface, color, [P_prev[0],P_prev[1]], [P[0],P[1]], width)
        if dist_arrow!=0:
            dist = dist + np.sqrt((x-x_prev)**2 + (y-y_prev)**2)
            x_prev = x
            y_prev = y
            if dist>=dist_arrow:
                if dt>0:
                    draw_direction(surface, P, 10, np.arctan2(P[1]-P_prev[1],P[0]-P_prev[0]), color)
                else:
                   draw_direction(surface, P, 10, np.arctan2(P_prev[1]-P[1],P_prev[0]-P[0]), color)
                dist = 0
        P_prev=P
        t = t +dt

################################################################################################################################################################
# Método de Runge Kutta-4 modificado para soporte de "tiempos negativos" 
################################################################################################################################################################

def plot_rk4(surface, color, ta, tb, dt, x0, y0, width, dist_arrow):
    if ta>tb:
        temp = ta
        ta = tb
        tb = temp

    plot_rk4_pos(surface, color, ta, tb, dt, x0, y0, width, dist_arrow)
    if ta<0:
         plot_rk4_pos(surface, color, tb, ta, -dt, x0, y0, width, dist_arrow)


################################################################################################################################################################
# Inicio del programa 
################################################################################################################################################################
plot_grid(screen, vp2d, 1, 1, "grey")
plot_axis(screen,vp2d)

pygame.display.flip()

################################################################################################################################################################
# Bucle de eventos 
################################################################################################################################################################
while True:
    pygame.draw.rect(screen, "white", pygame.Rect(0, screen_height-40, 800, screen_height))
    x, y = pygame.mouse.get_pos()
    P = project_2d_inv(vp2d, np.array([x,y]))
    type = analize_point(P,False)
    draw_text(screen, 0, screen_height-40, "( "+ "{:2.2f}".format(P[0]) + " , " + "{:2.2f}".format(P[1]) + " )" + " "+type, "black")        
    pygame.display.flip()

    flag=""
    for ev in pygame.event.get():
        if ev.type == QUIT:
            pygame.quit()
            sys.exit()
                       
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                x, y = pygame.mouse.get_pos()
                P = project_2d_inv(vp2d, np.array([x,y]))
                plot_rk4(screen, "blue", 0, 100, 0.001, P[0], P[1], 2, 0.5)
                pygame.display.flip()
            if ev.button ==3:
                x, y = pygame.mouse.get_pos()
                P = project_2d_inv(vp2d, np.array([x,y]))
                analize_point(P,True)
                
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_e:
                screen.fill((255, 255, 255))
            if ev.key == pygame.K_a:
               plot_grid(screen, vp2d, 1, 1, "dimgrey")
               plot_axis(screen,vp2d)
               
            if ev.key == pygame.K_p:
                flag="plot"
            if ev.key == pygame.K_s:
                flag="slope"
              
    if flag=="plot":
        y0 = ya
        for i in range(ny+1):
            x0 = xa
            for j in range(nx+1):
                plot_rk4(screen, "blue", 0, 10, 0.001, x0, y0, 2, 0.25)

                x0 = x0 + dx
            y0 = y0 + dy
    elif flag=="slope":
        s_max=0
        y0 = ya
        for i in range(ny+1):
            x0 = xa
            for j in range(nx+1):
                s = np.sqrt(x0**2+y0**2)
                if s>s_max:
                    s_max=s
                x0 = x0 + dx
            y0 = y0 + dy

        y0 = ya
        for i in range(ny+1):
            x0 = xa
            for j in range(nx+1):
                #plot_slope(screen, vp2d, np.array([x0,y0]),  dy_dt(x0, y0),  dx_dt(x0, y0), (0,128,0), 10, 2)       
                plot_slope2(screen, vp2d, np.array([x0,y0]),  dy_dt(x0, y0),  dx_dt(x0, y0), 20, 2,s_max)       
                x0 = x0 + dx
            y0 = y0 + dy
