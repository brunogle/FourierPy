import xml.etree.ElementTree as ET
import cv2
from svgpathtools import svg2paths, wsvg
from svgpathtools import parse_path, Line, Path, wsvg
from cv2 import VideoWriter, VideoWriter_fourcc
import cmath
import numpy as np
import math

svg_path = 'utn.svg'       #Archivo SVG a leer

video_path = 'utn.avi'     #Archivo de salida de video

codec = 'DM4V'              #Codigo de codec
FPS = 60                    #FPS del video generado

max_terms_seq = [50, 1000]     #Terminos a usar para cada segmento del video. -1 para usar todos #TODO:-1
durations_seq = [5, 10]       #Duraciones de cada segmento del video


point_density = 0.2                 #Densidad de puntos muestreados
vector_color = (122, 122, 122)            #Color de las lineas de los vectores de fondo (B, G, R)
circle_color = (100, 100, 100)            #Color de los circulos (B, G, R)
background_color = (255, 255, 255)  #Color de fondo





class Curves:
    def __init__(self, point_density):
        self.curves = []
        self.width = 0
        self.height = 0
        self.point_density = point_density
    
    def load_curves_from_svg(self, file):
        svg_tree = ET.parse(file)
        svg_root = svg_tree.getroot()
        self.width = int(float(svg_root.attrib['viewBox'].split()[2]))
        self.height = int(float(svg_root.attrib['viewBox'].split()[3]))

        for xml_path in svg_root.iter('{http://www.w3.org/2000/svg}path'):
           
            path = parse_path(xml_path.attrib['d'])

            path_len = path.length(error = 0.00001)

            path_points = []

            d = xml_path.attrib['d'][1:].split(',')

            num_points = int(path_len*self.point_density)           
            for i in range(num_points):
                path_points.append(path.point(i/num_points))

            
            css_class = '.' + xml_path.attrib['class']
            style_str = svg_root[0].text
            splited_styles = style_str.split(';}')
            color = list(substr for substr in splited_styles if substr.find(css_class) != -1)[0].split('stroke:#')[1][:6]
            color = tuple(int(color[i:i+2], 16) for i in (4, 2, 0)) #FORMAT: (B, G, R)


            self.curves.append(Curve(path_points, color))

    def calc_ffts(self):
        for c in self.curves:
            c.calc_fft()

class Phasor:
    def __init__(self, n, c):
        self.amplitude = abs(c)
        self.frequency = n
        self.phase = cmath.phase(c)

    def get(self, t):
        return cmath.rect(self.amplitude, self.phase + self.frequency*t)

class Curve:
    def __init__(self, curve_points, color):
        self.curve_points = curve_points
        self.color = color
        self.phasors = []
        self.drawn_points = []
        self.k = 0
        self.prev_k = 0

    def calc_fft(self):
        coefficients = np.fft.fft(self.curve_points)/len(self.curve_points)
        n = 0  
        for coef in coefficients:
            self.phasors.append(Phasor(n, coef))
            n += 1
        self.phasors = sorted(self.phasors, reverse=True, key=lambda x : x.amplitude)

class Renderer:

    def __init__(self, curves, file, codec, fps):
        self.curves = curves
        self.max_terms_seq = None
        self.durations_seq = None
        self.fourcc = VideoWriter_fourcc(*codec)
        self.video = VideoWriter('./' + file, self.fourcc, float(fps), (curves.width, curves.height))
        self.fps = fps
        self.width = curves.width
        self.height = curves.height

    def set_terms_and_durations(self, max_terms_seq, durations_seq):
        self.max_terms_seq = max_terms_seq
        self.durations_seq = durations_seq



    def render(self):

        total_rames_to_render = sum(self.durations_seq*self.fps)
        frames_rendered = 0

        for n_seq in range(len(self.max_terms_seq)):#RENDER SECUENCES
            max_terms = self.max_terms_seq[n_seq]
            num_frames = self.durations_seq[n_seq]*self.fps
            
            frame = None

            for curve in self.curves.curves:
                curve.drawn_points = []

            for frame_number in range(1, num_frames + 1):#RENDER FRAMES
                frames_rendered += 1
                if(frame_number%10 == 0):
                    print("Rendering... N = " + str(max_terms) + "(" + str(n_seq+1) + "/" + str(len(self.max_terms_seq)) + ")  FRAME "
                    + str(frame_number) + "/" + str(num_frames) + " (" + str(int(100*frames_rendered/total_rames_to_render))
                    + "%)", end="\r")

                frame = np.ones((self.height, self.width, 3), np.uint8)
                
                frame[:] = background_color
                
                for curve in self.curves.curves:#RENDER CURVES
                    
                    k = int(len(curve.phasors)*(frame_number - 1)/num_frames)

                    old_z = curve.phasors[0].get(0)
                 
                    for n in range(1, min(max_terms, len(curve.phasors))):
                        z = old_z + curve.phasors[n].get(2*math.pi*k/len(curve.phasors))
                        if(n < 100):
                            cv2.circle(frame, (int(old_z.real), int(old_z.imag)), int(curve.phasors[n].amplitude), circle_color, 1, lineType=cv2.LINE_AA)
                            cv2.line(frame, (int(old_z.real), int(old_z.imag)), (int(z.real), int(z.imag)), vector_color, 1, lineType=cv2.LINE_AA)
                        old_z = z
                    if(k > curve.prev_k + 1):
                        for k_aux in range(curve.prev_k+1, k):
                            z_aux = curve.phasors[0].get(0)
                            for n in range(1,min(max_terms, len(curve.phasors))):
                                z_aux += curve.phasors[n].get(2*math.pi*k_aux/len(curve.phasors))
                            curve.drawn_points.append([int(z_aux.real), int(z_aux.imag)])
                    curve.prev_k = k
                    curve.drawn_points.append([int(z.real), int(z.imag)])
        
                for curve in self.curves.curves:
                    if(len(curve.drawn_points) >= 2):
                        cv2.polylines(frame, np.int32([curve.drawn_points]), False, curve.color, 2, lineType=cv2.LINE_AA)
                
                cv2.putText(frame, 'N = ' + str(max_terms), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.video.write(frame)
            
            frame[:] = background_color
            for curve in self.curves.curves:
                if(len(curve.drawn_points) >= 2):
                    cv2.polylines(frame, np.int32([curve.drawn_points]), False, curve.color, 2, lineType=cv2.LINE_AA)
            
            cv2.putText(frame, 'N = ' + str(max_terms), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA)
            for i in range(self.fps*2):
                self.video.write(frame)
            
        self.video.release()
        print('')

curves = Curves(point_density)
curves.load_curves_from_svg(svg_path)
curves.calc_ffts()
renderer = Renderer(curves, video_path, codec, FPS)
renderer.set_terms_and_durations(max_terms_seq, durations_seq)
renderer.render()
