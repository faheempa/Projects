import random
import secrets
from classes import *


def main():
    size = width, height = (1200, 800)
    FPS = 60
    running = True
    GREEN = (0, 200, 0)
    ENEMY = "ENEMY_SHIP.png"
    level = 1
    enemy_start = True
    eno = 5
    player_speed = 5
    enemy_speed = 1
    player_reload = 0.5
    player_bullet_speed = 10
    ablity_timer=60*15
    next_level=0
    a=-1
    health_bar=600

    pygame.init()
    main_font = pygame.font.SysFont("monospace", 30)
    WIN = pygame.display.set_mode(size)
    pygame.display.set_caption("SPACE SHOOTER")
    WIN.fill(GREEN)

    BG = Image("bg.png", width / 2, height / 2, width, height)

    Player = Player_ship(
        "PLAYER_SHIP.png", width / 2, height / 2, player_reload, player_bullet_speed
    )
    Player.set_width_height(100, 100)
    Player.set_movement_border(0, 0, width, height)

    def create_enemies(n):
        Enemy_ship.active=n
        while n > 0:
            side = int(n % 4)
            if side == 0:
                x1, x2, y1, y2 = -500, 0, 0, height
            elif side == 1:
                x1, x2, y1, y2 = 0, width, -500, 0
            elif side == 2:
                x1, x2, y1, y2 = width, width + 500, 0, height
            elif side == 3:
                x1, x2, y1, y2 = 0, width, height, height + 500
            x = random.randrange(x1, x2)
            y = random.randrange(y1, y2)
            if side == 2:
                x1, x2, y1, y2 = -500, 0, 0, height
            elif side == 3:
                x1, x2, y1, y2 = 0, width, -500, 0
            elif side == 0:
                x1, x2, y1, y2 = width, width + 500, 0, height
            elif side == 1:
                x1, x2, y1, y2 = 0, width, height, height + 500
            gox = random.randrange(x1, x2)
            goy = random.randrange(y1, y2)
            r = secrets.choice([3, 4, 5, 6])
            e = Enemy_ship(ENEMY, x, y, gox, goy, enemy_speed, r, width, height)
            e.set_width_height(75, 75)
            Enemy_ship.enemy_list.append(e)
            n -= 1

    def redraw():
        BG.draw(WIN)
        Enemy_ship.draw_enemies_and_bullets(WIN)
        Player.draw_bullets(WIN)
        Player.draw(WIN)
        label_score = main_font.render(f"Score : {Player.score}", 1, (255, 255, 255))
        label_level = main_font.render(f"Level : {level}", 1, (255, 255, 255))
        WIN.blit(label_score, (10, 10))
        WIN.blit(label_level, (width - label_level.get_width() - 10, 10))
        pygame.draw.rect(WIN,(255,0,0),((width-health_bar)/2,15,health_bar,20))
        pygame.draw.rect(WIN,(0,255,0),((width-health_bar)/2,15,(Player.health/1000)*health_bar,20))
    
        pygame.display.update()

    def special_ablities(player):
        ablity=secrets.choice([1,2,3])
        if ablity==1:
            if player.health==1000:
                ablity=special_ablities(player)
            else:
                player.health_ablity()
                print("health")
        elif ablity==2:
            player.rage_ablity()
            print("rage")
        elif ablity==3:
            player.spin_ablity()
            print("spin")
        return ablity

    clock = pygame.time.Clock()
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEBUTTONDOWN and a!=3:
                Player.turn()

        if Player.health <= 0:
            print("Game Over")
            running = False
            continue

        Player.shoot()
        Enemy_ship.enemy_shot(Player)

        if enemy_start:
            create_enemies(eno)
            eno += 2
            enemy_start = False
        if Enemy_ship.active==0:
            Enemy_ship.active=-1
            next_level=300
            level+=1
        if next_level>0:
            next_level-=1
            if next_level==0:
                enemy_start=True

        keys = pygame.key.get_pressed()
        if keys[K_a] or keys[K_LEFT]:
            Player.move(-player_speed, 0)
        if keys[K_w] or keys[K_UP]:
            Player.move(0, -player_speed)
        if keys[K_d] or keys[K_RIGHT]:
            Player.move(player_speed, 0)
        if keys[K_s] or keys[K_DOWN]:
            Player.move(0, player_speed)

        if ablity_timer >0:
            ablity_timer-=1
            if ablity_timer==0:
                a = special_ablities(Player)
                ablity_timer=60*15

        if a==1:
            a=-1
        if a == 2:
            Player.rage_timer-=1
            if Player.rage_timer==0:
                a=-1
                Player.rage_ablity_end()
        if a == 3:
            Player.angle=(Player.angle+20)%360
            Player.rotate_at_angle(Player.angle)
            Player.spin_ablity_timer-=1
            if Player.spin_ablity_timer==0:
                a=-1
                Player.spin_ablity_end()
                
   
        Enemy_ship.move_enemies()
        Player.player_hit_enemy(Enemy_ship.enemy_list)
        Enemy_ship.enemy_hit_player(Player)
        Enemy_ship.remove_enemies()
        redraw()
    pygame.quit()


main()
