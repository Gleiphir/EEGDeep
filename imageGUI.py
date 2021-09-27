import pygame
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

image = Image.new("RGB", (640, 480))

draw = ImageDraw.Draw(image)

# points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
draw.polygon((points), fill=200)

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

pygame.init()
window = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()


pygameSurface = pilImageToSurface(image)

run = True
while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    window.fill(0)
    window.blit(pygameSurface, pygameSurface.get_rect(center = (320, 240)))
    pygame.display.flip()
