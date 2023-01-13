import pygame
from pygame.locals import *


class Image:
    def __init__(self, file_name, x, y, width=100, height=100) -> None:
        self.file_name = file_name
        self.width = width
        self.height = height
        self.loc = self.x, self.y = x, y
        self.img = pygame.image.load(self.file_name)
        self.img = pygame.transform.scale(self.img, (self.width, self.height))
        self.mask = pygame.mask.from_surface(self.img)
        self.moving = False
        self.have_movement_border = False
        self.draw_real = True
        self.draw_rotate = False
        self.angle = 0
        self.exist = True

    def draw(self, screen):
        if self.exist:
            if self.draw_real:
                self.img_rect = self.img.get_rect()
                self.img_rect.center = self.loc
                screen.blit(self.img, self.img_rect)
            if self.draw_rotate:
                self.img_rect = self.rotated_img.get_rect()
                self.img_rect.center = self.loc
                screen.blit(self.rotated_img, self.img_rect)

    def set_width_height(self, width, height):
        self.width = width
        self.height = height
        self.img = pygame.transform.scale(self.img, (self.width, self.height))

    def set_x(self, x):
        self.x = x
        self.set_loc((self.x, self.y))

    def set_y(self, y):
        self.y = y
        self.set_loc((self.x, self.y))

    def set_loc(self, pos):
        self.x, self.y = pos
        self.loc = (self.x, self.y)

    def is_inside(self, pos):
        x, y = pos
        if (
            x > self.x - self.width / 2
            and y > self.y - self.height / 2
            and x < self.x + self.width / 2
            and y < self.y + self.height / 2
        ):
            return True
        return False

    def is_outside(self, pos):
        x, y = pos
        if (
            x > self.x - self.width / 2
            and y > self.y - self.height / 2
            and x < self.x + self.width / 2
            and y < self.y + self.height / 2
        ):
            return False
        return True

    def move_using_mouse(self, event):
        if event.type == MOUSEBUTTONDOWN and self.is_inside(pygame.mouse.get_pos()):
            self.moving = True
        if event.type == MOUSEBUTTONUP:
            self.moving = False
        if self.moving and event.type == MOUSEMOTION:
            new_loc = pygame.mouse.get_pos()
            if self.allow_movement(new_loc):
                self.set_loc(new_loc)

    def set_movement_border(self, x1, y1, x2, y2):
        self.border_x1 = x1
        self.border_x2 = x2
        self.border_y1 = y1
        self.border_y2 = y2
        self.have_movement_border = True

    def allow_movement(self, pos):
        x, y = pos
        if self.have_movement_border:
            if (
                x - self.width / 2 > self.border_x1
                and x + self.width / 2 < self.border_x2
                and y - self.height / 2 > self.border_y1
                and y + self.height / 2 < self.border_y2
            ):
                return True
            self.moving = False
            return False
        return True

    def rotate_at_angle(self, angle):
        rotated_img_temp = pygame.transform.rotate(self.img, angle)
        self.width, self.height = (
            rotated_img_temp.get_width(),
            rotated_img_temp.get_height(),
        )
        if self.allow_movement(self.loc):
            self.angle = angle
            self.rotated_img=rotated_img_temp
            self.draw_real, self.draw_rotate = False, True
        else:
            self.width, self.height = (
                self.rotated_img.get_width(),
                self.rotated_img.get_height(),
            )

    def revert_rotations(self):
        self.draw_real, self.draw_rotate = True, False

    def move(self, x=0, y=0):
        new_x = self.x+x
        new_y = self.y+y
        if self.allow_movement((new_x, new_y)):
            self.set_loc((new_x, new_y))
            return True
        else:
            return False

    def remove(self):
        self.exist = False

    def collide_with(self,obj):
        offset_x = obj.x - self.x
        offset_y = obj.y - self.y
        return (self.mask.overlap(obj.mask, (offset_x, offset_y)) != None)
