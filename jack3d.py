from math import sqrt, pi
import numpy as np

TIME_STEP = 0.1
MIN_TIME = 0.
MAX_TIME = 100.

TIME_LIGHT_ARRAY = [0.] * int((MAX_TIME - MIN_TIME) / TIME_STEP)

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distanceToPoint(self, otherPoint):
        return sqrt((self.x-otherPoint.x)*(self.x-otherPoint.x) \
                    +(self.y-otherPoint.y)*(self.y-otherPoint.y) \
                    +(self.z-otherPoint.z)*(self.z-otherPoint.z))

    def distanceToWall(self, wall):
        return abs(self.x*wall.a + self.y*wall.b + self.z*wall.c + wall.d) / \
                sqrt(wall.a*wall.a + wall.b*wall.b + wall.c*wall.c)

class Wall:
    def __init__(self, a, b, c, d, unitVector1, unitVector2):
        # The wall is a plane characterized by the equation ax+by+cz=d
        self.a = a
        self.b = b
        self.c = c
        self.d = d
                
        # The two unit vectors form a basis that spans the wall        
        self.unitVector1 = np.array(unitVector1)
        self.unitVector2 = np.array(unitVector2)     
           
           
    def getListOfPoints(squareRadius, exampleVector):
        self.listOfPoints = []
        
        adjustedSquareRadius = int(squareRadius/SPACE_EPS)
        
        for uvFactor1Large in range(-adjustedSquareRadius, adjustedSquareRadius+1):
            uvFactor1 = uvFactor1Large*SPACE_EPS
            
            for uvFactor2Large in range(-adjustedSquareRadius, adjustedSquareRadius+1):
                uvFactor2 = uvFactor2Large*SPACE_EPS
                
                resultVector = uvFactor1*self.unitVector1 + \
                    uvFactor2*self.unitVector2 +
                    exampleVector
                
                self.listOfPoints.append(Point(resultVector[0], resultVector[1], 
                    resultVector[2]))
               
def addToTimeLightArray(time, lightAmount):
    timeIndex = int((time-MIN_TIME)/TIME_STEP)
    
    if timeIndex < len(TIME_LIGHT_ARRAY):
        TIME_LIGHT_ARRAY[timeIndex] += lightAmount               
    
def plotTimeLightArray():
    p.clf()
    p.plot([i*TIME_STEP for i in range(int(MIN_TIME/TIME_STEP), int(MAX_TIME/TIME_STEP))], TIME_LIGHT_ARRAY)
    p.savefig("tla.png")
    p.show()    
    
def lightFactor(distance, comingFromWall=False, receivedByWall=False, 
    distanceToWall=None):
    
    result = SPACE_EPS*SPACE_EPS/(4*pi*distance*distance)
    
    if receivedByWall:
        result *= distanceToWall/distance
        
    if comingFromWall:
        result *= 2
        
    return result

def timeOfLeg(sourcePoint, bouncePoint):
    return sourcePoint.distanceToPoint(bouncePoint)

# Light Factor functions for the primary experiment

def lightFactorFirstLeg(sourcePoint, bouncePoint):
    return 1
    
def lightFactorSecondLegPointTarget(bouncePoint, targetPoint):    
    return lightFactor(bouncePoint.distanceToPoint(targetPoint), comingFromWall=True,                           receivedByWall=False)

def lightFactorSecondLegWallTarget(bouncePoint, targetPoint, targetWall):
    return lightFactor(bouncePoint.distanceToPoint(targetPoint), comingFromWall=True,
            receivedByWall=True, bouncePoint.distanceToWall(targetWall))
    
def lightFactorThirdLegPointTarget(targetPoint, wallPoint, wall):
    return lightFactor(targetPoint.distanceToPoint(wallPoint), comingFromWall=False, 
            receivedByWall=True, targetPoint.distanceToWall(wall))
            
def lightFactorThirdLegWallTarget(targetPoint, wallPoint, wall):
    return lightFactor(targetPoint.distanceToPoint(wallPoint), comingFromWall=True,
            receivedByWall=True, targetPoint.distanceToWall(wall))
            
def lightFactorFourthLeg(wallPoint, detectorPoint):
    return lightFactor(wallPoint.distanceToPoint(detectorPoint), comingFromWall=True,
            receivedByWall=False)
            
# Light factor functions for the sphere-point experiment

def lightFactorSpherePoint(sourcePoint, wallPoint, wall):
    return lightFactor(sourcePoint.distanceToPoint(wallPoint), comingFromWall=False,
            receivedByWall=True, sourcePoint.distanceToWall(wall))
            
def doSpherePointExperiment():    
    sourcePoint = Point(0,0,1)
        
    wall = Wall(0,0,1,0, \
        np.array([1,0,0]), \
        np.array([0,1,0]))
            
    wallPoints = self.getListOfPoints(5, np.array([0,0,0]))
    
    for wallPoint in wallPoints:    
        time = timeOfLeg(sourcePoint, wallPoint)
        lightAmount = lightFactorSpherePoint(sourcePoint, wallPoint, wall)
        
        addToTimeLightArray(time, lightAmount)
        
        plotTimeLightArray()
    