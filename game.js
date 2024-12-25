// copyright 2025 by moshix and hotdog studios 
//up to now just create functional game
// 0.01 humble beginnings
// 0.01 - 1.00 just create functional game first
// 1.1  time based animation 
// 1.2  limit fire rate   
// 1.3  enemies also shoot!
// 1.4  player hit and lives system  
// 1.5  sound!
// 1.6  walls a bit bigger
// 1.6  use a vax to shoot at IBM
// 1.7  MUTE button
// 1.8  remove console log messages
// 1.9  1000 extra points when user finishes the level
// 2.0  touch screen support!
// 2.1  oh no! homing missiles from the evil empire!
// 2.2  fix missile logic, add assets and add favicon support
// 2.3  fix opening page and walls can now collapse! 
// 2.4  vax bullets also deterioate walls 
        
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const PLAYER_LIVES = 5; // Programmer tunable parameter for starting lives
const BULLET_SPEED = 300; // Player bullet speed (pixels per second)
const ENEMY_BULLET_SPEED = BULLET_SPEED / 3; // Enemy bullet speed (1/3 of player bullet speed)
const HIT_MESSAGE_DURATION = 1000; // How long to show "HIT!" message in milliseconds
const PLAYER_HIT_ANIMATION_DURATION = 1500; // Duration in milliseconds (1.5 seconds)
const MIN_MISSILE_INTERVAL = 3000; // 3 seconds
const MAX_MISSILE_INTERVAL = 6000; // 6 seconds
const MISSILE_SPEED = 200; // pixels per second
let lastMissileTime = 0;
let nextMissileTime = 0;
let homingMissiles = [];
let missileImage = new Image();
missileImage.src = 'missile.svg';

let player = {
  x: canvas.width / 2 - 25,
  y: canvas.height - 60,
  width: 50,
  height: 50,
  dx: 5,
  lives: PLAYER_LIVES,
  image: new Image(),
};
player.image.src = "vax.jpg";

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
let lastTime = 0;
const PLAYER_SPEED = 300; // pixels per second
const ENEMY_SPEED = 50; // pixels per second
const FIRE_RATE = 0.1; // Time in seconds between shots (0.1 = 10 shots per second)
const ENEMY_FIRE_RATE = 1.0; // Time in seconds between enemy shots
let lastEnemyFireTime = 0;

let hitMessageTimer = 0;
let showHitMessage = false;

let playerHitTimer = 0;
let isPlayerHit = false;
let playerNormalImage = new Image();
let playerExplosionImage = new Image();

playerNormalImage.src = "vax.svg";
playerExplosionImage.src = "explosion_player.jpg";
player.image = playerNormalImage;

const keys = {
  ArrowLeft: false,
  ArrowRight: false,
  Space: false,
  P: false,
  p: false,
  R: false,
  r: false,
};

let wallImage = new Image();
wallImage.src = 'wall.svg';
let chunkImage = new Image();
chunkImage.src = 'chunk.svg';

const WALL_MAX_HITS = 3;  // Programmer tunable: number of hits before wall disappears
const WALL_HITS_FROM_BELOW = 4;     // Programmer tunable: hits needed for wall damage from player shots
const WALL_MAX_HITS_TOTAL = 10;     // Programmer tunable: total hits before wall disappears
const WALL_MAX_MISSILE_HITS = 3;    // Programmer tunable: hits from missiles before wall disappears

let walls = [
  {
    x: canvas.width / 6 - 50,
    y: canvas.height - 150,
    width: 100,
    height: 20,
    image: wallImage
  },
  {
    x: canvas.width * 2/6 - 50,
    y: canvas.height - 150,
    width: 100,
    height: 20,
    image: wallImage
  },
  {
    x: canvas.width * 3/6 - 50,
    y: canvas.height - 150,
    width: 100,
    height: 20,
    image: wallImage
  },
  {
    x: canvas.width * 4/6 - 50,
    y: canvas.height - 150,
    width: 100,
    height: 20,
    image: wallImage
  },
  {
    x: canvas.width * 5/6 - 50,
    y: canvas.height - 150,
    width: 100,
    height: 20,
    image: wallImage
  }
];

// Initialize wallHits array for all walls
let wallHits = walls.map(() => []);

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
  if (isPlayerHit) {
    if (Date.now() - playerHitTimer > PLAYER_HIT_ANIMATION_DURATION) {
      isPlayerHit = false;
      player.image = playerNormalImage;
      player.width = 50;
      player.height = 50;
    } else {
      player.image = playerExplosionImage;
      player.width = 50 * 2;
      player.height = 50 * 2;
      // Only draw if image is loaded
      if (player.image.complete) {
        ctx.drawImage(
          player.image, 
          player.x - (player.width / 2),
          player.y - (player.height / 2),
          player.width, 
          player.height
        );
      }
      return;
    }
  }
  // Only draw if image is loaded
  if (player.image.complete) {
    ctx.drawImage(player.image, player.x, player.y, player.width, player.height);
  }
}

function drawEnemies() {
  enemies.forEach((enemy) => {
    if (enemy.image.complete) {
      ctx.drawImage(enemy.image, enemy.x, enemy.y, enemy.width, enemy.height);
    } else {
      ctx.fillStyle = 'red';
      ctx.fillRect(enemy.x, enemy.y, enemy.width, enemy.height);
      enemy.image.onload = () => {
        ctx.drawImage(enemy.image, enemy.x, enemy.y, enemy.width, enemy.height);
      };
    }
  });
}

function drawBullets() {
  bullets.forEach((bullet) => {
    if (bullet.isEnemyBullet) {
      ctx.fillStyle = "red";  // Enemy bullets are red
    } else {
      ctx.fillStyle = "white"; // Player bullets remain white
    }
    ctx.fillRect(bullet.x, bullet.y, 5, 10);
  });
}

function drawWalls() {
  walls.forEach((wall, index) => {
    // Draw wall SVG first
    if (wall.image.complete) {
      ctx.drawImage(wall.image, wall.x, wall.y, wall.width, wall.height);
    }
    
    // Draw damage chunks on top
    if (wallHits[index]) {
      wallHits[index].forEach(hit => {
        if (chunkImage.complete) {
          ctx.drawImage(
            chunkImage,
            wall.x + hit.x - 10, // Center chunk on hit location
            wall.y + hit.y - 10,
            20,  // chunk size
            20
          );
        }
      });
    }
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

function movePlayer(deltaTime) {
  // Only allow movement if player is not in hit animation
  if (!isPlayerHit) {
    if (keys.ArrowLeft && player.x > 0) {
      player.x -= PLAYER_SPEED * deltaTime;
    }
    if (keys.ArrowRight && player.x < canvas.width - player.width) {
      player.x += PLAYER_SPEED * deltaTime;
    }
  }
}

function moveBullets(deltaTime) {
  bullets.forEach((bullet) => {
    if (bullet.isEnemyBullet) {
      bullet.y += ENEMY_BULLET_SPEED * deltaTime; // Slower enemy bullets
    } else {
      bullet.y -= BULLET_SPEED * deltaTime;
    }
  });
  bullets = bullets.filter((bullet) => 
    bullet.y > 0 && bullet.y < canvas.height
  );
}

function moveEnemies(deltaTime) {
  const currentEnemySpeed = (ENEMY_SPEED * enemySpeed) * deltaTime;  // Combine base speed with level multiplier
  enemies.forEach((enemy) => {
    enemy.x += currentEnemySpeed * enemyDirection;
    if (enemy.x + enemy.width > canvas.width || enemy.x < 0) {
      enemyDirection *= -1;
      enemies.forEach((e) => (e.y += e.height * 0.25));
    }
    if (enemy.y + enemy.height >= walls[0].y - 100) {
      gameOverFlag = true;
    }
  });
  enemies = enemies.filter((enemy) => enemy.y < canvas.height);
  if (enemies.length === 0 && !gameOverFlag) {
    victoryFlag = true;
  }
}

function detectCollisions() {
  // Check player bullet collisions with walls
  bullets.forEach((bullet, bIndex) => {
    if (!bullet.isEnemyBullet) {  // Only player bullets
      walls.forEach((wall, wallIndex) => {
        if (bullet.x >= wall.x && 
            bullet.x <= wall.x + wall.width &&
            bullet.y >= wall.y && 
            bullet.y <= wall.y + wall.height) {
          
          // Count hits for this wall
          wall.hitCount = (wall.hitCount || 0) + 1;
          
          // Add damage mark every WALL_HITS_FROM_BELOW shots
          if (wall.hitCount % WALL_HITS_FROM_BELOW === 0) {
            wallHits[wallIndex].push({
              x: bullet.x - wall.x,
              y: bullet.y - wall.y
            });
          }
          
          // Remove bullet
          bullets.splice(bIndex, 1);
          
          // Remove wall if total hits exceeded
          if (wall.hitCount >= WALL_MAX_HITS_TOTAL) {
            walls.splice(wallIndex, 1);
            wallHits.splice(wallIndex, 1);
          }
        }
      });
    }
  });

  // Update missile collision with walls to use WALL_MAX_MISSILE_HITS
  homingMissiles.forEach((missile, mIndex) => {
    walls.forEach((wall, wallIndex) => {
      if (missile.x >= wall.x && 
          missile.x <= wall.x + wall.width &&
          missile.y >= wall.y && 
          missile.y <= wall.y + wall.height) {
        
        wallHits[wallIndex].push({
          x: missile.x - wall.x,
          y: missile.y - wall.y
        });
        
        homingMissiles.splice(mIndex, 1);
        
        // Count missile hits separately
        wall.missileHits = (wall.missileHits || 0) + 1;
        if (wall.missileHits >= WALL_MAX_MISSILE_HITS) {
          walls.splice(wallIndex, 1);
          wallHits.splice(wallIndex, 1);
        }
      }
    });
  });

  for (let bIndex = bullets.length - 1; bIndex >= 0; bIndex--) {
    const bullet = bullets[bIndex];
    
    if (!bullet.isEnemyBullet) {
      for (let mIndex = homingMissiles.length - 1; mIndex >= 0; mIndex--) {
        const missile = homingMissiles[mIndex];
        const dx = bullet.x - missile.x;
        const dy = bullet.y - missile.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < (missile.width/2 + 5)) {
          // Add explosion at missile's position
          missileExplosions.push({
            x: missile.x,
            y: missile.y,
            width: missile.width * 1.5,  // Make explosion slightly larger
            height: missile.width * 1.5,
            timeCreated: Date.now(),
            duration: 400  // Match sound duration
          });
          
          bullets.splice(bIndex, 1);
          homingMissiles.splice(mIndex, 1);
          
          missileBoomSound.currentTime = 0;
          if (!isMuted) {
            missileBoomSound.play();
            setTimeout(() => {
              missileBoomSound.pause();
              missileBoomSound.currentTime = 0;
            }, 400);
          }
          break;
        }
      }
    }
  }
  bullets.forEach((bullet, bIndex) => {
    if (bullet.isEnemyBullet) {
      // Only check player collision if not currently hit
      if (!isPlayerHit) {
        if (bullet.x < player.x + player.width &&
            bullet.x + 5 > player.x &&
            bullet.y < player.y + player.height &&
            bullet.y + 10 > player.y) {
          bullets.splice(bIndex, 1);
          player.lives--;
          showHitMessage = true;
          hitMessageTimer = Date.now();
          
          // Add explosion animation and sound
          isPlayerHit = true;
          playerHitTimer = Date.now();
          player.image = playerExplosionImage;
          playerExplosionSound.currentTime = 0; // Reset sound to start
          playerExplosionSound.play();
          
          // Clear all enemy bullets
          bullets = bullets.filter(b => !b.isEnemyBullet);
          
          if (player.lives <= 0) {
            gameOverFlag = true;
            gameOverSound.currentTime = 0;
            gameOverSound.play();
          }
        }
      }
    } else {
      // Existing collision detection for player bullets
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
    }

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

  // Check missile collisions with player
  homingMissiles.forEach((missile, mIndex) => {
    if (!isPlayerHit) {
      const dx = (missile.x) - (player.x + player.width/2);
      const dy = (missile.y) - (player.y + player.height/2);
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < (player.width + missile.width)/2) {
        homingMissiles.splice(mIndex, 1);
        player.lives--;
        showHitMessage = true;
        hitMessageTimer = Date.now();
        
        // Add explosion animation and sound
        isPlayerHit = true;
        playerHitTimer = Date.now();
        player.image = playerExplosionImage;
        playerExplosionSound.currentTime = 0;
        playerExplosionSound.play();
        
        // Clear all enemy bullets and missiles
        bullets = bullets.filter(b => !b.isEnemyBullet);
        homingMissiles = [];
        
        if (player.lives <= 0) {
          gameOverFlag = true;
        }
      }
    }
  });
}

function drawScore() {
  ctx.save();
  ctx.fillStyle = "white";
  ctx.font = "16px Arial";
  ctx.textAlign = "left";
  document.getElementById("score").innerText = 
    `Score: ${score}\nLives: ${player.lives}/${PLAYER_LIVES}\nLevel: ${currentLevel}`;
  ctx.restore();
}

function gameOver() {
  ctx.fillStyle = "white";
  ctx.font = "50px Arial";
  ctx.textAlign = "center";
  ctx.fillText("You lost! Game Over!", canvas.width / 2, canvas.height / 2);
  gamePaused = true;
}

function victory() {
  currentLevel++;
  enemySpeed *= 1.33;
  score += 1000;  // Add 1000 points for completing the level
  enemies = [];
  createEnemies();
  bullets = [];
  gamePaused = false;
  victoryFlag = false;
  lastTime = 0;
  requestAnimationFrame(gameLoop);
}

function restartGame() {
  currentLevel = 1;
  enemySpeed = 0.55;
  player.lives = PLAYER_LIVES;
  player.x = canvas.width / 2;
  player.y = canvas.height - 60;
  player.image = playerNormalImage;
  isPlayerHit = false;
  score = 0;
  bullets = [];
  enemies = [];
  homingMissiles = [];
  createEnemies();
  explosions = [];
  showHitMessage = false;
  gamePaused = false;
  gameOverFlag = false;
  victoryFlag = false;
  lastTime = 0;
  startGameSound.currentTime = 0;
  startGameSound.play();
  requestAnimationFrame(gameLoop);
  missileExplosions = [];
  walls = [
    {
      x: canvas.width / 6 - 50,
      y: canvas.height - 150,
      width: 100,
      height: 20,
      image: wallImage
    },
    {
      x: canvas.width * 2/6 - 50,
      y: canvas.height - 150,
      width: 100,
      height: 20,
      image: wallImage
    },
    {
      x: canvas.width * 3/6 - 50,
      y: canvas.height - 150,
      width: 100,
      height: 20,
      image: wallImage
    },
    {
      x: canvas.width * 4/6 - 50,
      y: canvas.height - 150,
      width: 100,
      height: 20,
      image: wallImage
    },
    {
      x: canvas.width * 5/6 - 50,
      y: canvas.height - 150,
      width: 100,
      height: 20,
      image: wallImage
    }
  ].map(wall => ({...wall, hitCount: 0, missileHits: 0}));
  wallHits = walls.map(() => []);
}

function handleEnemyShooting(currentTime) {
  if (currentTime - lastEnemyFireTime < ENEMY_FIRE_RATE * 1000) return;
  
  // Find the lowest enemy in each column
  const lowestEnemies = [];
  enemies.forEach(enemy => {
    const columnIndex = Math.floor(enemy.x / (enemy.width + 20)); // Using padding of 20
    if (!lowestEnemies[columnIndex] || enemy.y > lowestEnemies[columnIndex].y) {
      lowestEnemies[columnIndex] = enemy;
    }
  });
  
  // Find the enemy closest to player among the lowest enemies
  const closestEnemy = lowestEnemies.reduce((closest, enemy) => {
    if (!closest) return enemy;
    if (!enemy) return closest;
    return Math.abs(enemy.x - player.x) < Math.abs(closest.x - player.x) ? enemy : closest;
  }, null);
  
  // Fire bullet from closest enemy
  if (closestEnemy) {
    bullets.push({
      x: closestEnemy.x + closestEnemy.width / 2,
      y: closestEnemy.y + closestEnemy.height,
      dy: -5, // Negative because enemy bullets move down
      isEnemyBullet: true
    });
    lastEnemyFireTime = currentTime;
  }
}

function drawHitMessage() {
  if (showHitMessage) {
    ctx.save();
    ctx.fillStyle = "red";
    ctx.font = "bold 48px Arial";
    ctx.textAlign = "center";
    ctx.fillText("HIT!", canvas.width / 2, canvas.height / 2);
    ctx.restore();
    
    if (Date.now() - hitMessageTimer > HIT_MESSAGE_DURATION) {
      showHitMessage = false;
    }
  }
}

function drawMuteStatus() {
  if (isMuted) {
    ctx.save();
    ctx.fillStyle = "white";
    ctx.font = "16px Arial";
    ctx.textAlign = "right";
    ctx.fillText("MUTED", canvas.width - 260, canvas.height - 10);  // Position below version number
    ctx.restore();
  }
}

function drawPauseMessage() {
  if (gamePaused) {
    ctx.save();
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "48px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("PAUSED", canvas.width/2, canvas.height/2);
    ctx.restore();
  }
}

function gameLoop(currentTime) {
  if (!lastTime) lastTime = currentTime;
  const deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw everything regardless of pause state
  drawPlayer();
  drawEnemies();
  drawBullets();
  drawMissiles();
  drawMissileExplosions();
  drawWalls();
  drawExplosions();
  drawScore();
  drawHitMessage();
  drawMuteStatus();
  drawPauseMessage();  // Add pause message last so it's on top

  // Only update positions if game is not paused
  if (!gamePaused) {
    if (player.lives > 0 && !gameOverFlag && !victoryFlag) {
      movePlayer(deltaTime);
      moveBullets(deltaTime);
      moveEnemies(deltaTime);
      moveMissiles(deltaTime);
      handleEnemyShooting(currentTime);
      handleMissileLaunching(currentTime);
      detectCollisions();
    }

    if (gameOverFlag) {
      gameOver();
    } else if (victoryFlag) {
      victory();
    }
  }

  if (!gameOverFlag && !victoryFlag) {
    requestAnimationFrame(gameLoop);
  }
}

function startGame() {
  document.getElementById("legend").style.display = "none";
  enemies = []; // Clear any existing enemies
  createEnemies();
  lastTime = 0; // Reset the time
  startGameSound.currentTime = 0;
  startGameSound.play();
  requestAnimationFrame(gameLoop);
}

document.addEventListener("keydown", (e) => {
  if (e.code === "KeyP") {
    gamePaused = !gamePaused;
    if (!gamePaused) {
      lastTime = 0;
      requestAnimationFrame(gameLoop);
    }
  }
  if (e.code === "KeyR") {
    restartGame();
  }
  if (e.code === "Space" && !gamePaused && 
      Date.now() - lastFireTime > FIRE_RATE * 1000) {
    if (!keys.Space) {
      spaceKeyPressTime = Date.now();
    }
    
    bullets.push({
      x: player.x + player.width / 2 - 2.5,
      y: player.y,
      isEnemyBullet: false
    });
    lastFireTime = Date.now();
    
    // Play appropriate sound based on hold duration
    if (Date.now() - spaceKeyPressTime > MACHINE_GUN_THRESHOLD) {
      if (!isMuted) {
        machineGunSound.currentTime = 0;
        machineGunSound.play();
        // Clear any existing timer
        if (machineGunSoundTimer) clearTimeout(machineGunSoundTimer);
        // Set new timer to stop sound after duration
        machineGunSoundTimer = setTimeout(() => {
          machineGunSound.pause();
          machineGunSound.currentTime = 0;
        }, machineGunSoundDuration);
      }
    } else {
      if (!isMuted) {
        playerShotSound.currentTime = 0;
        playerShotSound.play();
      }
    }
  }
  if (e.code === "KeyM") {
    isMuted = !isMuted;
    // Mute/unmute all sounds except startgame
    playerExplosionSound.muted = isMuted;
    gameOverSound.muted = isMuted;
  }
  if (e.code in keys) keys[e.code] = true;
});

document.addEventListener("keyup", (e) => {
  if (e.code in keys) {
    keys[e.code] = false;
    if (e.code === "Space") {
      spaceKeyPressTime = 0; // Reset the space key timer
      machineGunSound.pause(); // Stop the machine gun sound
      machineGunSound.currentTime = 0; // Reset the sound to start
    }
  }
});

document.addEventListener("keydown", startGame, { once: true });

let playerExplosionSound = new Audio('playerhit.mp3');
let startGameSound = new Audio('startgame.mp3');
let gameOverSound = new Audio('overgame.mp3');

let isMuted = false;

let playerShotSound = new Audio('playershot3.mp3');
let machineGunSound = new Audio('mgun.mp3');
let spaceKeyPressTime = 0;
const MACHINE_GUN_THRESHOLD = 500; // 0.5 seconds in milliseconds
let machineGunSoundDuration = 500; // 0.5 seconds in milliseconds
let machineGunSoundTimer = null;

let currentLevel = 1;

// Add function to get random enemy from top row
function getRandomTopRowEnemy() {
  const topY = Math.min(...enemies.map(e => e.y));
  const topRowEnemies = enemies.filter(e => e.y === topY);
  return topRowEnemies[Math.floor(Math.random() * topRowEnemies.length)];
}

// Add function to handle missile launching
function handleMissileLaunching(currentTime) {
  if (currentTime >= nextMissileTime) {
    const shooter = getRandomTopRowEnemy();
    if (shooter) {
      homingMissiles.push({
        x: shooter.x + shooter.width/2,
        y: shooter.y + shooter.height,
        angle: 0,
        width: 30,
        height: 30,
        time: 0 // For trajectory calculation
      });
    }
    // Set next missile time
    nextMissileTime = currentTime + 
      Math.random() * (MAX_MISSILE_INTERVAL - MIN_MISSILE_INTERVAL) + 
      MIN_MISSILE_INTERVAL;
  }
}

// Add function to move missiles
function moveMissiles(deltaTime) {
  homingMissiles.forEach(missile => {
    missile.time += deltaTime;
    
    // Check if missile is above wall row
    const isAboveWallRow = missile.y < (walls[0].y - 50); // One row above walls
    
    if (isAboveWallRow) {
      // Calculate target direction
      const dx = player.x + player.width/2 - missile.x;
      const dy = player.y + player.height/2 - missile.y;
      
      // Calculate missile angle
      missile.angle = Math.atan2(dy, dx);
    }
    // else keep the last angle
    
    // Add curved trajectory
    const curve = Math.sin(missile.time * 2) * 100;
    
    // Move missile
    missile.x += Math.cos(missile.angle) * MISSILE_SPEED * deltaTime;
    missile.y += Math.sin(missile.angle) * MISSILE_SPEED * deltaTime;
    missile.x += Math.cos(missile.angle + Math.PI/2) * curve * deltaTime;
  });
  
  // Remove missiles that are off screen
  homingMissiles = homingMissiles.filter(m => 
    m.y < canvas.height && m.y > 0 && m.x > 0 && m.x < canvas.width
  );
}

// Add function to draw missiles
function drawMissiles() {
  homingMissiles.forEach(missile => {
    ctx.save();
    ctx.translate(missile.x, missile.y);
    ctx.rotate(missile.angle + Math.PI/2); // Rotate missile to face direction of travel
    ctx.drawImage(
      missileImage,
      -missile.width/2,
      -missile.height/2,
      missile.width,
      missile.height
    );
    ctx.restore();
  });
}

// Add with other audio elements at the top
let missileBoomSound = new Audio('explode_missile.mp3');

// Add with other image declarations at the top
let missileExplosionImage = new Image();
missileExplosionImage.src = 'explode_missile.jpg';

// Add missile explosions array with other state variables
let missileExplosions = [];

// Add function to draw missile explosions
function drawMissileExplosions() {
  missileExplosions = missileExplosions.filter(explosion => {
    const age = Date.now() - explosion.timeCreated;
    if (age < explosion.duration) {
      ctx.drawImage(
        missileExplosionImage,
        explosion.x - explosion.width/2,
        explosion.y - explosion.height/2,
        explosion.width,
        explosion.height
      );
      return true;
    }
    return false;
  });
}
