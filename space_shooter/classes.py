from cmath import atan, cos
from math import radians, degrees
from multiprocessing import set_forkserver_preload
import numpy as np
from image_class import *


class Ship(Image):
    def __init__(self, file_name, x, y, health=100) -> None:
        super().__init__(file_name, x, y)
        self.health = health
        self.bullets = []

    def draw_bullets(self, win):
        for bullet in self.bullets:
            bullet.draw(win)
            if bullet.can_move == False:
                self.bullets.remove(bullet)
                bullet.remove()
                del bullet
            else:
                bullet.move_bullet()


class Player_ship(Ship):
    def __init__(self, file_name, x, y, reload_time, bullet_speed) -> None:
        super().__init__(file_name, x, y)
        self.reload = 0
        self.health = 1000
        self.reload_time = reload_time
        self.bullet_speed=bullet_speed
        self.score=0
        self.rage_timer=0
        self.spin_ablity_timer=0
        

    def turn(self):
        mx, my = pygame.mouse.get_pos()
        x, y = mx - self.x, -(my - self.y)
        try:
            tangent_value = round(y / x, 2)
            angle = degrees(atan(tangent_value).real)
            if mx < self.x:
                angle = 180 + angle
            self.rotate_at_angle(angle)
        except:
            pass

    def shoot(self):
        if self.reload == 0:
            a = Bullet(
                "PLAYER_BULLET.png", self.x, self.y, 20, 20, 50, self.bullet_speed, self.reload_time
            )
            a.set_movement_border(
                self.border_x1, self.border_y1, self.border_x2, self.border_y2
            )
            self.bullets.append(a)
            a.aquire_xy_to_shoot(self.angle)
            a.rotate_at_angle(self.angle)
            self.reload = self.reload_time * 60
        else:
            self.reload -= 1

    def player_hit_enemy(self, enemies):
        for b in self.bullets:
            for e in enemies:
                if b.collide_with(e) and b.exist:
                    self.score+=1
                    e.remove()
                    b.remove()

    def health_ablity(self):
        self.health+=500
        if self.health>1000:
            self.health=1000

    def rage_ablity(self):
        self.reload_time=0.1
        self.rage_timer=60*8

    def rage_ablity_end(self):
        self.reload_time=0.5

    def spin_ablity(self):
        self.spin_ablity_timer=60*1.5
        self.reload_time=0

    def spin_ablity_end(self):
        self.reload_time=0.5

class Enemy_ship(Ship):
    enemy_list = []
    active=0
    def __init__(
        self, file_name, x, y, x2, y2, speed, reload_time, width, height
    ) -> None:
        super().__init__(file_name, x, y)
        self.reload = 0
        self.x2 = x2
        self.y2 = y2
        self.speed = speed
        self.reload_time = reload_time
        self.screen_width = width
        self.screen_height = height
        self.inside = False
        self.find_direction()
        self.find_speed()
        self.bullet_name="ENEMY_BULLET.png"

    def check_status(self):
        if self.inside == False:
            if (
                self.x > 0
                and self.x < self.screen_width
                and self.y > 0
                and self.y < self.screen_height
            ):
                self.inside = True
        else:
            if (
                self.x < 0
                or self.x > self.screen_width
                or self.y < 0
                or self.y > self.screen_width
            ):
                self.inside = False
                self.remove()

    def find_direction(self):
        x, y = self.x2 - self.x, -(self.y2 - self.y)
        try:
            tangent_value = round(y / x, 2)
            self.angle = degrees(atan(tangent_value).real)
            if self.x2 < self.x:
                self.angle = 180 + self.angle
        except:
            return

    def find_speed(self):
        def simplify(fraction):
            y, x = fraction
            while (
                x > self.speed / 2
                or y > self.speed / 2
                or x < -self.speed / 2
                or y < -self.speed / 2
            ):
                x = x / 4
                y = y / 4
            return (y, x)

        def resultant(x, y):
            a = abs(x)
            b = abs(y)
            return (a**2 + b**2 + a * b * cos(radians(self.angle)).real) ** 0.5

        def amplify(fraction):
            y, x = fraction
            while resultant(x, y) < self.speed:
                x *= 1.1
                y *= 1.1
            return (y, x)

        rad_angle = radians(self.angle)
        tangent = round(np.tan(rad_angle).real, 3)
        self.sy, self.sx = amplify(simplify(tangent.as_integer_ratio()))
        if self.angle < 90:
            self.sy = -self.sy
            self.sx = -self.sx
        if self.angle > 270:
            self.sy = -self.sy
            self.sx = self.sx
        else:
            self.sx = -self.sx

    def move_enemy(self):
        if self.exist:
            self.ok = self.move(self.sx, self.sy)

    def shoot(self, player):
        if self.inside:
            if self.reload == 0:
                blt = Bullet(
                    self.bullet_name, self.x, self.y, 30, 30, 100, 4, self.reload_time
                )
                blt.x2, blt.y2 = player.loc
                blt.find_angle()
                self.bullets.append(blt)
                self.reload = self.reload_time * 60
            else:
                self.reload -= 1

    @staticmethod
    def remove_enemies():
        for e in Enemy_ship.enemy_list:
            e.check_status()
            if e.exist == False:
                Enemy_ship.enemy_list.remove(e)
                del e
                if Enemy_ship.active>0:
                    Enemy_ship.active-=1
                    
    @staticmethod
    def draw_enemies_and_bullets(win):
        for enemy in Enemy_ship.enemy_list:
            enemy.draw(win)
            enemy.draw_bullets(win)

    @staticmethod
    def move_enemies():
        for enemy in Enemy_ship.enemy_list:
            enemy.move_enemy()

    @staticmethod
    def enemy_shot(player):
        for e in Enemy_ship.enemy_list:
            e.shoot(player)

    @staticmethod
    def enemy_hit_player(player):
        for e in Enemy_ship.enemy_list:
            for b in e.bullets:
                if b.collide_with(player) and b.exist:
                    b.remove()
                    player.health-=100
            if e.collide_with(player):
                e.remove()
                player.health-=100

class Bullet(Image):
    def __init__(
        self, file_name, x, y, width, height, damage, speed, reload_time
    ) -> None:
        super().__init__(file_name, x, y, width, height)
        self.damage = damage
        self.can_move = True
        self.speed = speed
        self.reload_time = reload_time

    def find_angle(self):
        x, y = self.x2 - self.x, -(self.y2 - self.y)
        try:
            tangent_value = round(y / x, 2)
            self.angle = degrees(atan(tangent_value).real)
            if self.x2 < self.x:
                self.angle = 180 + self.angle
        except:
            return
        self.aquire_xy_to_shoot(self.angle)

    def aquire_xy_to_shoot(self, angle):
        def simplify(fraction):
            y, x = fraction
            while (
                x > self.speed / 2
                or y > self.speed / 2
                or x < -self.speed / 2
                or y < -self.speed / 2
            ):
                x = x / 4
                y = y / 4
            return (y, x)

        def resultant(x, y):
            a = abs(x)
            b = abs(y)
            return (a**2 + b**2 + a * b * cos(radians(angle)).real) ** 0.5

        def amplify(fraction):
            y, x = fraction
            while resultant(x, y) < self.speed:
                x *= 1.1
                y *= 1.1
            return (y, x)

        rad_angle = radians(angle)
        tangent = round(np.tan(rad_angle).real, 3)
        self.sy, self.sx = amplify(simplify(tangent.as_integer_ratio()))
        if angle < 90:
            self.sy = -self.sy
            self.sx = -self.sx
        if angle > 270:
            self.sy = -self.sy
            self.sx = self.sx
        else:
            self.sx = -self.sx

    def move_bullet(self):
        self.can_move = self.move(self.sx, self.sy)


