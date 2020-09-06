import cv2

class SimplePreprocessor:
     def __init__(self, width, height, inter=cv2.INTER_AREA):
          # store the target image width, height, and interpolation
          # method used when resizing
          self.width = width
          self.height = height
          self.inter = inter
          
     def preprocess(self, image):
          # resize the image to a fixed size, ignoring the aspect
          # ratio
     #      return cv2.resize(image, (self.width, self.height),
     #           interpolation=self.inter)
               
     # def resize(self,image):
	# initialize the dimensions of the image to be resized and
	# grab the image size
          dim = None
          (h, w) = image.shape[:2]

          # if both the width and height are None, then return the
          # original image
          if self.width is None and self.height is None:
               return image

          # check to see if the width is None
          if self.width is None:
               # calculate the ratio of the height and construct the
               # dimensions
               r = self.height / float(h)
               dim = (int(w * r), self.height)

          # otherwise, the height is None
          else:
               # calculate the ratio of the width and construct the
               # dimensions
               r = self.width / float(w)
               dim = (self.width, int(h * r))

          dim = (self.width, self.height)

          # resize the image
          resized = cv2.resize(image, dim, interpolation = self.inter)
          resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

          # return the resized image
          return resized