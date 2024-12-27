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
// 2.0  touch screen support! (later removed again)
// 2.1  oh no! homing missiles from the evil empire!
// 2.2  fix missile logic, add assets and add favicon support
// 2.3  fix opening page and walls can now collapse! 
// 2.4  vax bullets also deterioate walls 
// 2.5  fix some sound issues 
// 2.5.1 fix new level unexplained pause
// 2.5.2 also enable A for left and D for right move
// 2.5.3 show new level banner before continuing
// 2.6   fire rate slower while moving  
// 2.7  avoid missed promise in sound playing
// 2.8  fix game over race condition
// 2.9  sometimes change explosion type for enemies to make more interesting
// 3.0  more sound effects
// 3.0.1 old bug of enemy reacching criticla position fix?
// 3.0.2 display game over when enmies reach critical position
// 3.0.3 aha! (?) only enemies alive can reach wall
// 3.1   catch bug when no more walls are around 
// 3.2   defintely show game over when it's over
// 3.2.1 small parameter tune-ups    
// 3.2.2 enemies fire more frequent as levels increase 
// 3.2.3 version taken from the javascript file
// 3.24  change enemy explosions graphics a bit
// 3.2.5 clode clean up 
// 3.3   touch support again !
// 3.4   columns of enemies respond to browser window size 
// 3.5.1 touch controls now also fire bullets
// 3.6   monster on top of screen
// 3.6.1 the monster can shoot! 
// 3.6.2 make monster shoot missiles from its position  
    
const VERSION = "v3.6.1";  // version showing in index.html


document.getElementById('version-info').textContent = VERSION;

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const PLAYER_LIVES = 5;    // starting lives
const BULLET_SPEED = 300; // Player bullet speed (pixels per second)
const ENEMY_BULLET_SPEED = BULLET_SPEED / 3; // Enemy bullet speed (1/3 of player bullet speed)
const HIT_MESSAGE_DURATION = 1000; // How long to show "HIT!" message in milliseconds
const PLAYER_HIT_ANIMATION_DURATION = 1500; // Duration in milliseconds (1.5 seconds)
const MIN_MISSILE_INTERVAL = 3000; // 3 seconds
const MAX_MISSILE_INTERVAL = 6700; // 6 seconds
const MISSILE_SPEED = 170; // pixels per second

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


let bullets = [];
let enemies = [];
let explosions = [];
let score = 0;
let enemyHitsToDestroy = 2; // how many times an enemy needs to be hit
let enemySpeed = 0.45;  // Decreased from 0.55
let enemyDirection = 1; // 1 for right, -1 for left
let gamePaused = false;
let lastFireTime = 0;
let gameOverFlag = false;
let victoryFlag = false;
let lastTime = 0;
const PLAYER_SPEED = 300; // pixels per second
const ENEMY_SPEED = 50; // pixels per second
const FIRE_RATE = 0.2; // Time in seconds between shots (0.1 = 10 shots per second)
const ENEMY_FIRE_RATE = 0.9; // Time in seconds between enemy shots
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
  KeyD: false,
  KeyA: false,
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

const WALL_HITS_FROM_BELOW = 3;  // hits needed for wall damage from player shots
const WALL_MAX_HITS_TOTAL = 13;  // total hits before wall disappears
const WALL_MAX_MISSILE_HITS = 4; // hits from missiles before wall disappears

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

//counter for tracking explosions
let explosionCounter = 0;

// At the start of the game where other assets are loaded
let explosionImg = new Image();
explosionImg.src = 'explosion.svg';
let explosionAdditionalImg = new Image();
explosionAdditionalImg.src = 'explosion_additional.svg';

const BASE_FIRE_RATE = 0.2;           // Base time in seconds between shots
let currentFireRate = BASE_FIRE_RATE; // Current fire rate that can be modified

// for enemies  
const BASE_ENEMY_FIRE_RATE = 0.85;    // Base time in seconds between enemy shots
const ENEMY_FIRE_RATE_INCREASE = 0.15;// 15% increase per level
let currentEnemyFireRate = BASE_ENEMY_FIRE_RATE;

// Add near the top with other initialization code
let isTouchDevice = false;
let isTablet = false;

// Detect if device is a tablet
function detectTablet() {
  // Check if touch device
  isTouchDevice = ('ontouchstart' in window) || 
                 (navigator.maxTouchPoints > 0) || 
                 (navigator.msMaxTouchPoints > 0);
  
  // Check if tablet based on screen size
  isTablet = isTouchDevice && 
             Math.min(window.innerWidth, window.innerHeight) >= 768 && 
             Math.max(window.innerWidth, window.innerHeight) <= 1366;
             
  // Show/hide touch controls based on device
  const touchControls = document.getElementById('touch-controls');
  if (touchControls) {
    touchControls.style.display = isTablet ? 'block' : 'none';
  }
}

// Initialize touch controls
function initTouchControls() {
  detectTablet();
  
  if (!isTablet) return;

  const touchStart = document.getElementById('touch-start');
  const touchLeft = document.getElementById('touch-left');
  const touchRight = document.getElementById('touch-right');
  const touchFireLeft = document.getElementById('touch-fire-left');
  const touchFireRight = document.getElementById('touch-fire-right');

  // Start button
  touchStart.addEventListener('click', () => {
    startGame();
    touchStart.style.display = 'none';
  });

  // Movement controls
  touchLeft.addEventListener('touchstart', (e) => {
    e.preventDefault();
    keys.ArrowLeft = true;
  });

  touchLeft.addEventListener('touchend', () => {
    keys.ArrowLeft = false;
  });

  touchRight.addEventListener('touchstart', (e) => {
    e.preventDefault();
    keys.ArrowRight = true;
  });

  touchRight.addEventListener('touchend', () => {
    keys.ArrowRight = false;
  });

  // Modified fire controls
  function handleFireStart(e) {
    e.preventDefault();
    keys.Space = true;
    spaceKeyPressTime = Date.now();
    // Add direct bullet creation here
    if (!gamePaused && Date.now() - lastFireTime > currentFireRate * 1000) {
      bullets.push({
        x: player.x + player.width / 2 - 2.5,
        y: player.y,
        isEnemyBullet: false
      });
      lastFireTime = Date.now();
      playSoundWithCleanup(() => playerShotSound);
    }
  }

  function handleFireEnd(e) {
    e.preventDefault();
    keys.Space = false;
    stopMachineGunSound();
  }

  // Add both touchstart and click events for better response
  touchFireLeft.addEventListener('touchstart', handleFireStart);
  touchFireLeft.addEventListener('touchend', handleFireEnd);
  touchFireRight.addEventListener('touchstart', handleFireStart);
  touchFireRight.addEventListener('touchend', handleFireEnd);
}

// Add resize handler to update tablet detection
window.addEventListener('resize', detectTablet);

// Initialize touch controls when the window loads
window.addEventListener('load', initTouchControls);

function createExplosion(x, y) {
  explosionCounter++;
  if (explosionCounter % 2 === 0) {
    playSoundWithCleanup(createExplosionSound);
  }
  score += 30;

  const isAdditionalExplosion = explosionCounter % (Math.random() < 0.5 ? 2 : 3) === 0;
  
  const newExplosion = {
    x: x,
    y: y,
    frame: 0,
    img: new Image(),
    width: isAdditionalExplosion ? 170 : 96,   // Regular explosion now 96 (32 * 3)
    height: isAdditionalExplosion ? 170 : 96,  // Regular explosion now 96 (32 * 3)
  };
  
  // Set the source first, then push to array
  newExplosion.img.src = isAdditionalExplosion ? 'explosion_additional.svg' : 'explosion.svg';
  explosions.push(newExplosion);
}

function createEnemies() {
    const rows = 5;
    // Calculate number of columns based on window width
    const minCols = 4;  // Minimum number of columns
    const maxCols = 12; // Maximum number of columns
    const enemyWidth = 58;
    const padding = 20;
    const minTotalWidth = (enemyWidth + padding) * minCols;
    
    // Calculate how many columns can fit in the current window width
    let cols = Math.floor((canvas.width - 60) / (enemyWidth + padding)); // 60 is total side padding
    cols = Math.max(minCols, Math.min(maxCols, cols)); // Clamp between min and max
    
    const enemyHeight = 58;
    const offsetTop = 30;
    // Center the enemies horizontally
    const offsetLeft = (canvas.width - (cols * (enemyWidth + padding))) / 2;

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
      explosion.img,
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
    if ((keys.ArrowLeft || keys.KeyA) && player.x > 0) {
      player.x -= PLAYER_SPEED * deltaTime;
    }
    if ((keys.ArrowRight || keys.KeyD) && player.x < canvas.width - player.width) {
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
  // Add safety check for large deltaTime values
  if (deltaTime > 0.1) deltaTime = 0.1;  // Cap maximum deltaTime at 100ms
  
  const currentEnemySpeed = (ENEMY_SPEED * enemySpeed) * deltaTime;
  let needsToMoveDown = false;
  
  // First check if any enemy needs to change direction
  enemies.forEach((enemy) => {
    // Skip any "dead" enemies that might still be in the array
    if (enemy.hits >= enemyHitsToDestroy) return;
    
    enemy.x += currentEnemySpeed * enemyDirection;
    if (enemy.x + enemy.width > canvas.width || enemy.x < 0) {
      needsToMoveDown = true;
      enemy.x = Math.max(0, Math.min(canvas.width - enemy.width, enemy.x));
    }
  });
  
  // Then handle direction change and moving down as a separate step
  if (needsToMoveDown) {
    enemyDirection *= -1;
    const moveDownAmount = 20;
    enemies.forEach((enemy) => {
      // Only check active enemies
      if (enemy.hits < enemyHitsToDestroy) {
        enemy.y += moveDownAmount;
        if (enemy.y + enemy.height >= walls[0].y - 50) {
          console.log('Game Over triggered by ACTIVE enemy position:', enemy.y + enemy.height, 'wall position:', walls[0].y - 50);
          gameOverFlag = true;
          gameOver();
        }
      }
    });
  }
  
  // Clean up any destroyed enemies
  enemies = enemies.filter(enemy => enemy.hits < enemyHitsToDestroy);
  
  if (enemies.length === 0 && !gameOverFlag) {
    gamePaused = true;
    Object.keys(keys).forEach(key => {
        keys[key] = false;
    });
    victory();
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
            playSoundWithCleanup(createWallGoneSound);
            walls.splice(wallIndex, 1);
            wallHits.splice(wallIndex, 1);
          }
        }
      });
    }
  });

  // Update missile collision with walls
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

  // Bullet collisions with missiles
  for (let bIndex = bullets.length - 1; bIndex >= 0; bIndex--) {
    const bullet = bullets[bIndex];
    
    if (!bullet.isEnemyBullet) {
      for (let mIndex = homingMissiles.length - 1; mIndex >= 0; mIndex--) {
        const missile = homingMissiles[mIndex];
        const dx = bullet.x - missile.x;
        const dy = bullet.y - missile.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < (missile.width/2 + 5)) {
          missileExplosions.push({
            x: missile.x,
            y: missile.y,
            width: missile.width * 1.5,
            height: missile.width * 1.5,
            timeCreated: Date.now(),
            duration: 400
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

  // Enemy bullets and player collision
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
          playerExplosionSound.currentTime = 0;
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
      // Player bullets hitting enemies
      enemies.forEach((enemy, eIndex) => {
        if (bullet.x < enemy.x + enemy.width &&
            bullet.x + 5 > enemy.x &&
            bullet.y < enemy.y + enemy.height &&
            bullet.y + 10 > enemy.y) {
          enemy.hits++;
          if (enemy.hits >= enemyHitsToDestroy) {
            createExplosion(enemy.x, enemy.y);
            enemies.splice(eIndex, 1);
            score += 10;
          }
          bullets.splice(bIndex, 1);
        }
      });
    }
  });

  // Monster collision detection
  if (monster && !monster.hit) {
    bullets.forEach((bullet, bIndex) => {
      if (!bullet.isEnemyBullet) {
        // Check if there are any enemies directly in the bullet's path to the monster
        const hasEnemyInPath = enemies.some(enemy => 
          bullet.x >= enemy.x && 
          bullet.x <= enemy.x + enemy.width && 
          bullet.y > enemy.y && 
          bullet.y < monster.y + monster.height
        );

        // Only check monster collision if no enemies are in the way
        if (!hasEnemyInPath && 
            bullet.x < monster.x + monster.width &&
            bullet.x + 5 > monster.x &&
            bullet.y < monster.y + monster.height &&
            bullet.y + 10 > monster.y) {
          
          bullets.splice(bIndex, 1);
          monster.hit = true;
          monster.hitTime = Date.now();
          score += 1000;  // Bonus points for hitting monster
          
          // Play monster death sound
          if (!isMuted) {
            monsterDeadSound.currentTime = 0;
            monsterDeadSound.play();
          }
        }
      }
    });
  }
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
  enemySpeed *= 1.33;  // Existing speed increase
  
  // Increase enemy fire rate by reducing the time between shots
  currentEnemyFireRate = BASE_ENEMY_FIRE_RATE / 
      (1 + (ENEMY_FIRE_RATE_INCREASE * (currentLevel - 1)));
  
  score += 1000;  // Add 1000 points for completing the level
  enemies = [];
  bullets = [];
  
  // Draw the level message
  ctx.save();
  ctx.fillStyle = "white";
  ctx.font = "48px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`Level ${currentLevel}`, canvas.width/2, canvas.height/2);
  ctx.restore();
  
  // Wait 1.5 seconds before starting the new level
  setTimeout(() => {
    createEnemies();
    gamePaused = false;
    victoryFlag = false;
    lastTime = 0;
    requestAnimationFrame(gameLoop);
  }, 1500);
}

function restartGame() {
  currentLevel = 1;
  enemySpeed = 0.45;
  currentEnemyFireRate = BASE_ENEMY_FIRE_RATE;  // Reset enemy fire rate
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
  
  if (isTablet) {
    document.getElementById("touch-start").style.display = "block";
  }
}

function handleEnemyShooting(currentTime) {
  if (currentTime - lastEnemyFireTime < currentEnemyFireRate * 1000) return;
  
  // Find the lowest enemy in each column
  const lowestEnemies = [];
  enemies.forEach(enemy => {
    const columnIndex = Math.floor(enemy.x / (enemy.width + 20));
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
      dy: -5,
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
  if (gamePaused && !gameOverFlag && enemies.length > 0 && currentLevel === Math.floor(currentLevel)) {  // Only show PAUSED during regular gameplay
    ctx.save();
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "48px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("PAUSED", canvas.width/2, canvas.height/2);
    ctx.restore();
  }
}

function drawLevelMessage() {
  if (gamePaused && !gameOverFlag && enemies.length === 0) {  // Only show during level transition
    ctx.save();
    ctx.fillStyle = "white";
    ctx.font = "48px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`Level ${currentLevel}`, canvas.width/2, canvas.height/2);
    ctx.restore();
  }
}

function gameLoop(currentTime) {
  // Calculate deltaTime first
  if (!lastTime) {
    lastTime = currentTime;
  }
  const deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Handle shooting
  if (keys.Space && !gamePaused && 
      Date.now() - lastFireTime > currentFireRate * 1000) {
    bullets.push({
      x: player.x + player.width / 2 - 2.5,
      y: player.y,
      isEnemyBullet: false
    });
    lastFireTime = Date.now();
    playSoundWithCleanup(() => playerShotSound);
  }

  // Game logic
  if (!gamePaused && !gameOverFlag) {
    createMonster(currentTime);
    moveMonster(deltaTime);
  }

  // Draw everything
  drawPlayer();
  drawEnemies();
  drawMonster();  // Draw monster after enemies
  drawBullets();
  drawMissiles();
  drawMissileExplosions();
  drawWalls();
  drawExplosions();
  drawScore();
  drawHitMessage();
  drawMuteStatus();
  drawLevelMessage();
  drawPauseMessage();

  // Update game elements if not paused
  if (!gamePaused && !gameOverFlag) {
      movePlayer(deltaTime);
      if (player.lives > 0) {
          moveBullets(deltaTime);
          moveEnemies(deltaTime);
          moveMissiles(deltaTime);
          handleEnemyShooting(currentTime);
          handleMissileLaunching(currentTime);
          detectCollisions();
      }
  }

  // Draw game over message if needed
  if (gameOverFlag) {
      ctx.fillStyle = "white";
      ctx.font = "50px Arial";
      ctx.textAlign = "center";
      ctx.fillText("GAME OVER!", canvas.width / 2, canvas.height / 2);
  }

  requestAnimationFrame(gameLoop);
}

function startGame() {
  document.getElementById("legend").style.display = "none";
  if (isTablet) {
    document.getElementById("touch-start").style.display = "none";
  }
  enemies = [];
  createEnemies();
  lastTime = 0;
  startGameSound.currentTime = 0;
  startGameSound.play();
  requestAnimationFrame(gameLoop);
}

document.addEventListener("keydown", (e) => {
  if (e.code === "F11") {
    // Clear all enemies to trigger victory condition
    enemies = [];
    // Reset all key states
    Object.keys(keys).forEach(key => {
        keys[key] = false;
    });
    victory();
  }
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
  if (e.code in keys) {
    keys[e.code] = true;
    if (e.code === "Space" && !keys.Space) {
      spaceKeyPressTime = Date.now();
    }
  }
  if (e.code === "KeyM") {
    isMuted = !isMuted;
    // Mute/unmute all sounds except startgame
    playerExplosionSound.muted = isMuted;
    gameOverSound.muted = isMuted;
  }
});

document.addEventListener("keyup", (e) => {
  if (e.code in keys) {
    keys[e.code] = false;
    if (e.code === "Space") {
      spaceKeyPressTime = 0;
      stopMachineGunSound();
    }
  }
});

document.addEventListener("keydown", startGame, { once: true });

// mute control boolean
let isMuted = false;

let playerExplosionSound = new Audio('playerhit.mp3');
let startGameSound = new Audio('startgame.mp3');
let gameOverSound = new Audio('overgame.mp3');
let monsterDeadSound = new Audio('monster_dead.mp3');

let playerShotSound = new Audio('playershot3.mp3');
let machineGunSound = new Audio('mgun.mp3');
let spaceKeyPressTime = 0;
const MACHINE_GUN_THRESHOLD = 500; // 0.5 seconds in milliseconds



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
        width: 44,
        height: 44,
        time: 0 // For trajectory calculation
      });
    }
    newMissileLaunched = true;
    playSoundWithCleanup(createMissileLaunchSound);
    newMissileLaunched = false;
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
    
    // Check if there are any walls left before checking position
    const wallRowY = walls.length > 0 ? walls[0].y - 50 : canvas.height * 0.85;
    
    // Check if missile is above wall row
    const isAboveWallRow = missile.y < wallRowY;
    
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

// Update audio handling
function playSound(sound) {
    if (!isMuted) {
        // Reset the sound before playing
        sound.pause();
        sound.currentTime = 0;
        
        // Create a play promise
        const playPromise = sound.play();

        if (playPromise !== undefined) {
            playPromise.catch(error => {
                if (error.name === "AbortError") {
                    // Ignore abort errors - these happen when rapidly firing
                    console.log("Sound play aborted");
                } else {
                    console.error("Error playing sound:", error);
                }
            });
        }
    }
}

// Update where sounds are played
function handleShooting() {
    if (keys.Space && !gamePaused) {
        const currentTime = Date.now();
        if (currentTime - lastShootTime >= shootCooldown) {
            bullets.push(createBullet(player.x + player.width/2, player.y, false));
            playSound(shootSound);
            lastShootTime = currentTime;
        }
    }
}

// Sound creation functions
function createExplosionSound() {
    const sound = new Audio('explosion_enemy.mp3');
    sound.volume = 1.0;
    return sound;
}

function createWallGoneSound() {
    const sound = new Audio('wall_gone.mp3');
    sound.volume = 1.0;
    return sound;
}

function createMissileLaunchSound() {
    const sound = new Audio('missile_flying_short.mp3');
    sound.volume = 1.0;
    return sound;
}

// Example of how to play sounds (apply this pattern to all sound plays)
function playSoundWithCleanup(createSoundFunc) {
    if (!isMuted) {
        const sound = createSoundFunc();
        const playPromise = sound.play();

        if (playPromise !== undefined) {
            playPromise
                .then(() => {
                    // Sound played successfully, schedule cleanup
                    setTimeout(() => {
                        try {
                            sound.pause();
                            sound.remove();
                        } catch (e) {
                            console.log("Cleanup error handled:", e);
                        }
                    }, 1000);
                })
                .catch(error => {
                    if (error.name === "AbortError") {
                        // Ignore abort errors - these happen when rapidly firing
                        console.log("Sound play aborted - normal during rapid fire");
                    } else {
                        console.error("Error playing sound:", error);
                    }
                });
        }
    }
}

// Update how we handle machine gun sound
let machineGunSoundPlaying = false;

function playMachineGunSound() {
    if (!isMuted && !machineGunSoundPlaying) {
        machineGunSoundPlaying = true;
        machineGunSound.currentTime = 0;
        const playPromise = machineGunSound.play();
        
        if (playPromise !== undefined) {
            playPromise.catch(error => {
                if (error.name === "AbortError") {
                    console.log("Machine gun sound aborted");
                } else {
                    console.error("Error playing machine gun sound:", error);
                }
                machineGunSoundPlaying = false;
            });
        }
    }
}

function stopMachineGunSound() {
    if (machineGunSoundPlaying) {
        try {
            machineGunSound.pause();
            machineGunSound.currentTime = 0;
        } catch (e) {
            console.log("Error stopping machine gun sound:", e);
        }
        machineGunSoundPlaying = false;
    }
}

// Add near other state variables
let monster = null;
let monsterDirection = 1;  // 1 for right, -1 for left
let lastMonsterTime = 0;
const MONSTER_INTERVAL = 5000;  // 5 seconds between monster appearances
const MONSTER_SPEED = 200;      // pixels per second
const MONSTER_WIDTH = 73;       // monster size
const MONSTER_HEIGHT = 73;
const MONSTER_HIT_DURATION = 700;  // 0.7 seconds
let monsterHit = false;
let monsterImage = new Image();
let monsterHitImage = new Image();
monsterImage.src = 'monster.svg';
monsterHitImage.src = 'monster_shot.svg';

// Add this function to handle monster creation
function createMonster(currentTime) {
    if (!monster && currentTime - lastMonsterTime > MONSTER_INTERVAL) {
        monsterDirection = Math.random() < 0.5 ? 1 : -1;
        
        // Calculate starting position - start just off screen
        const startX = monsterDirection === 1 ? -MONSTER_WIDTH : canvas.width + MONSTER_WIDTH;
        const topEnemyRow = Math.min(...enemies.map(e => e.y)) - 50;
        
        monster = {
            x: startX,
            y: Math.max(topEnemyRow, MONSTER_HEIGHT),
            width: MONSTER_WIDTH,
            height: MONSTER_HEIGHT,
            hit: false,
            hitTime: 0,
            hasShot: false  // Add flag to track if monster has fired its missiles
        };
        
        lastMonsterTime = currentTime;
    }
}

// Add this function to move monster
function moveMonster(deltaTime) {
    if (monster) {
        if (monster.hit) {
            if (Date.now() - monster.hitTime > MONSTER_HIT_DURATION) {
                monster = null;
                lastMonsterTime = performance.now();
            }
        } else {
            // Move the monster
            monster.x += MONSTER_SPEED * monsterDirection * deltaTime;
            
            // Debug logging
          //  console.log("Monster position:", monster.x);
            console.log("Has shot:", monster.hasShot);
            console.log("Is on screen:", monster.x > 0 && monster.x + monster.width < canvas.width);
            
            // Check if monster is on screen and hasn't shot yet
            if (!monster.hasShot && 
                monster.x > 0 && 
                monster.x + monster.width < canvas.width) {
                
                console.log("FIRING MISSILES!");
                
                // Fire exactly 2 missiles
                for (let i = 0; i < 2; i++) {
                    const missile = {
                        x: monster.x + (monster.width/2),
                        y: monster.y + monster.height,
                        angle: Math.PI/2,  // Point downward
                        width: 44,
                        height: 44,
                        time: 0
                    };
                    homingMissiles.push(missile);
                    console.log("Missile created at:", missile.x, missile.y);
                }
                
                playSoundWithCleanup(createMissileLaunchSound);
                monster.hasShot = true;
            }
            
            // Check if monster has moved off screen
            if ((monsterDirection === 1 && monster.x > canvas.width + MONSTER_WIDTH) ||
                (monsterDirection === -1 && monster.x < -MONSTER_WIDTH)) {
                monster = null;
                lastMonsterTime = performance.now();
            }
        }
    }
}

// Add this function to draw monster
function drawMonster() {
    if (monster) {
        const image = monster.hit ? monsterHitImage : monsterImage;
        if (image.complete) {
            ctx.drawImage(image, monster.x, monster.y, monster.width, monster.height);
        }
    }
}

// Add sound for monster hit
function createMonsterHitSound() {
    const sound = new Audio('monster_hit.mp3');
    sound.volume = 1.0;
    return sound;
}

// Add with other monster-related constants at the top
const MONSTER_MISSILE_INTERVAL = 500; // Time between monster's missile shots (500ms)
let lastMonsterMissileTime = 0;




