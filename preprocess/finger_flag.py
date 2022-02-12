class Finger:
    def __int__(self):
        pass

    @staticmethod
    def SingleOne():
        thumb = False
        index = True
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleTwo():
        thumb = False
        index = True
        middle = True
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleThree():
        thumb = False
        index = True
        middle = True
        ring = True
        pinky = False
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleFour():
        thumb = False
        index = True
        middle = True
        ring = True
        pinky = True
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleFive():
        thumb = True
        index = True
        middle = True
        ring = True
        pinky = True
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleSix():
        thumb = True
        index = False
        middle = False
        ring = False
        pinky = True
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleSeven():
        thumb = True
        index = True
        middle = False
        ring = False
        pinky = True
        return thumb, index, middle, ring, pinky

    @staticmethod
    def SingleEight():
        thumb = True
        index = True
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky
