const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let player = {
  x: canvas.width / 2,
  y: canvas.height - 60,
  width: 50,
  height: 50,
  dx: 5,
  lives: 3,
  image: new Image(),
};
player.image.src = "hercules-ship.svg";

let bullets = [];
let enemies = [];
let explosions = [];
let score = 0;
let enemyHitsToDestroy = 2; // Configurable parameter
let enemySpeed = 0.55; // 33% faster speed for enemies
let enemyDirection = 1; // 1 for right, -1 for left
let gamePaused = false;
let lastFireTime = 0;
let gameOverFlag = false;
let victoryFlag = false;

const keys = {
  ArrowLeft: false,
  ArrowRight: false,
  Space: false,
  P: false,
  p: false,
  R: false,
  r: false,
};

let walls = [
  {
    x: canvas.width * 0.2 - 25,
    y: player.y - 50,
    width: 50,
    height: 30,
    image: new Image(),
  },
  {
    x: canvas.width * 0.4 - 25,
    y: player.y - 50,
    width: 50,
    height: 30,
    image: new Image(),
  },
  {
    x: canvas.width * 0.6 - 25,
    y: player.y - 50,
    width: 50,
    height: 30,
    image: new Image(),
  },
  {
    x: canvas.width * 0.8 - 25,
    y: player.y - 50,
    width: 50,
    height: 30,
    image: new Image(),
  },
];
walls.forEach((wall) => (wall.image.src = "wall.svg"));

function createEnemies() {
  const rows = 5;
  const cols = 10;
  const enemyWidth = 58; // Increased by 25% from 40 to 50
  const enemyHeight = 58; // Increased by 25% from 40 to 50
  const padding = 20;
  const offsetTop = 30;
  const offsetLeft = 30;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let color = ["red", "orange", "yellow", "green", "blue"][r % 5];
      enemies.push({
        x: c * (enemyWidth + padding) + offsetLeft,
        y: r * (enemyHeight + padding) + offsetTop,
        width: enemyWidth,
        height: enemyHeight,
        hits: 0,
        image: new Image(),
      });
      enemies[enemies.length - 1].image.src = `enemy-ship-${color}.svg`;
    }
  }
}

function drawPlayer() {
  ctx.drawImage(player.image, player.x, player.y, player.width, player.height);
}

function drawEnemies() {
  enemies.forEach((enemy) => {
    ctx.drawImage(enemy.image, enemy.x, enemy.y, enemy.width, enemy.height);
  });
}

function drawBullets() {
  ctx.fillStyle = "white";
  bullets.forEach((bullet) => {
    ctx.fillRect(bullet.x, bullet.y, 5, 10);
  });
}

function drawWalls() {
  walls.forEach((wall) => {
    ctx.drawImage(wall.image, wall.x, wall.y, wall.width, wall.height);
  });
}

function drawExplosions() {
  explosions.forEach((explosion) => {
    ctx.drawImage(
      explosion.image,
      explosion.x,
      explosion.y,
      explosion.width,
      explosion.height,
    );
  });
  explosions = explosions.filter(
    (explosion) => Date.now() - explosion.startTime < 500,
  ); // Remove after 500ms
}

function movePlayer() {
  if (keys.ArrowLeft && player.x > 0) {
    player.x -= player.dx;
  }
  if (keys.ArrowRight && player.x < canvas.width - player.width) {
    player.x += player.dx;
  }
}

function moveBullets() {
  bullets.forEach((bullet) => {
    bullet.y -= bullet.dy;
  });
  bullets = bullets.filter((bullet) => bullet.y > 0);
}

function moveEnemies() {
  enemies.forEach((enemy) => {
    enemy.x += enemySpeed * enemyDirection;
    if (enemy.x + enemy.width > canvas.width || enemy.x < 0) {
      enemyDirection *= -1;
      enemies.forEach((enemy) => (enemy.y += enemy.height * 1.25)); // 25% faster descent
    }
    if (enemy.y + enemy.height >= player.y - 50) {
      gameOverFlag = true;
    }
  });
  enemies = enemies.filter((enemy) => enemy.y < canvas.height);
  if (enemies.length === 0 && !gameOverFlag) {
    victoryFlag = true;
  }
}

function detectCollisions() {
  bullets.forEach((bullet, bIndex) => {
    enemies.forEach((enemy, eIndex) => {
      if (
        bullet.x < enemy.x + enemy.width &&
        bullet.x + 5 > enemy.x &&
        bullet.y < enemy.y + enemy.height &&
        bullet.y + 10 > enemy.y
      ) {
        enemy.hits++;
        if (enemy.hits >= enemyHitsToDestroy) {
          explosions.push({
            x: enemy.x,
            y: enemy.y,
            width: enemy.width,
            height: enemy.height,
            image: new Image(),
            startTime: Date.now(),
          });
          explosions[explosions.length - 1].image.src = "explosion.svg";
          enemies.splice(eIndex, 1);
          score += 10;
        }
        bullets.splice(bIndex, 1);
      }
    });

    walls.forEach((wall) => {
      if (
        bullet.x < wall.x + wall.width &&
        bullet.x + 5 > wall.x &&
        bullet.y < wall.y + wall.height &&
        bullet.y + 10 > wall.y
      ) {
        bullets.splice(bIndex, 1);
      }
    });
  });
}

function drawScore() {
  document.getElementById("score").innerText = `Score: ${score}`;
}

function gameOver() {
  ctx.fillStyle = "white";
  ctx.font = "50px Arial";
  ctx.fillText("Game Over", canvas.width / 2 - 150, canvas.height / 2);
  gamePaused = true;
}

function victory() {
  ctx.fillStyle = "white";
  ctx.font = "50px Arial";
  ctx.fillText(
    "You have destroyed the enemy!",
    canvas.width / 2 - 300,
    canvas.height / 2,
  );
  gamePaused = true;
}

function restartGame() {
  player.lives = 3;
  score = 0;
  bullets = [];
  enemies = [];
  explosions = [];
  createEnemies();
  gamePaused = false;
  gameOverFlag = false;
  victoryFlag = false;
  gameLoop();
}

function gameLoop() {
  if (keys.R || keys.r) {
    restartGame();
    keys.R = false;
    keys.r = false;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!gamePaused) {
    if (player.lives > 0 && !gameOverFlag && !victoryFlag) {
      movePlayer();
      moveBullets();
      moveEnemies();
      detectCollisions();

      drawPlayer();
      drawBullets();
      drawEnemies();
      drawWalls();
      drawExplosions();
      drawScore();

      if (gameOverFlag) {
        gameOver();
      } else if (victoryFlag) {
        victory();
      } else {
        requestAnimationFrame(gameLoop);
      }
    } else {
      if (gameOverFlag) {
        gameOver();
      } else if (victoryFlag) {
        victory();
      }
    }
  } else {
    requestAnimationFrame(gameLoop);
  }
}

function startGame() {
  document.getElementById("legend").style.display = "none";
  createEnemies();
  gameLoop();
}

document.addEventListener("keydown", (e) => {
  if (e.code in keys) keys[e.code] = true;
  if (
    e.code === "Space" &&
    player.lives > 0 &&
    Date.now() - lastFireTime > 50
  ) {
    // Limit fire rate to 1/20 sec
    bullets.push({
      x: player.x + player.width / 2 - 2.5,
      y: player.y,
      dy: 5,
    });
    lastFireTime = Date.now();
  }
  if (e.code === "P" || e.code === "p") {
    gamePaused = !gamePaused;
    if (!gamePaused) {
      gameLoop(); // Restart game loop if unpausing
    }
  }
  if (e.code === "R" || e.code === "r") {
    restartGame();
  }
});

document.addEventListener("keyup", (e) => {
  if (e.code in keys) keys[e.code] = false;
});

document.addEventListener("keydown", startGame, { once: true });
