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
// 3.6.3 restore walls when monster hit  
// 3.6.4 fix collision detection regression with missiles...arghhh   
// 3.6.5 Yannai fixed F11 race condition 
// 3.6.6 Small tune ups (due to monster death regenerating walls again) 
// 3.7   Show lives with space ships instead of just numbers 
// 3.7.1 Show brief animation in lower right corner when life is list
// 3.8   Every 5th shot down missile, player gets a bonus   
// 3.9   Firebirds opening screen, and walls protect from bullets 
// 4.0   Every 7 bonus grants, player gets one life back!
// 4.0.1 Small fixes in restart logic (reset values)
// 4.1   With new life, user is informed thru life grand animation 
// 4.2   Adjustements to canvas size, redo all html, and scale content
// 4.2.1 Some adjustements to positiongs and reset logic 
// 4.2.2 Put enemies a bit further down   
// 4.3   Make canvas always 1024x576, and center it on the screen
// 4.3.1 Scale up a bit more, and make bonusSound and new life grand silent if muted
// 4.4   AI mode with F1 !!! 
// 4.4.1 Refine bullet and missile avoidance  
// 4.4.2 better threat trajectory analysis  
// 4.4.3 better bullets avoidance in lateral movement for AI mode
// 4.4   also look sideways for bullets when moving in AI mode
// 4.5   Use space background image instead of solid color
// 4.5.1 Fix player size  
// 4.5.3 fix wall restoratin and re-initialization of game
// 4.5.4 wall damage handling
// 4.5.5 wall damage look nicer    
// 4.5.6 revised sounds 
// 4.6   first stable version with preloading of assets and sound
// 4.6.1 restrict certain locales, adjust monster position 
// 4.6.2 only advance speed of enemies by 9% between levels to make it more playable
// 4.6.3 limit enemy firing rate in new levels to make game more playable
// 4.6.4 make bullets a bit bigger
// 4.7   various playability improvements (no bullets while player is hit)     
// 4.8   kamikaze enemies! 
// 4.8.1 better kamikaze artwork 
// 4.9   hot streak message for player
// 4.9.1-6 fix various kamikaze small bugs
// 5.0   whole new game play! 
// 5.1   monster starts to move down at end of a sceneshouldSlalom = enemies.length < KAMIKA
// 5.2.1-5 fix monster slalom mod and ipad game playing issues
// 5.3   new monster enemy with different behavior patterns 
// 5.4   make a bit more playable and more monster2 patterns
// 5.5   code cleanup 
// 5.6   put enemy explosions back in, change points system a bit, code cleanup         

const VERSION = "v5.6.2g";  // version showing in index.html 

// keep right after the VERSION constant
if (document.getElementById('version-info')) {
    document.getElementById('version-info').textContent = VERSION;
}

// canvas size! 
const GAME_WIDTH = 1024;
const GAME_HEIGHT = 576;
 




const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

// Set the fixed canvas size
canvas.width = GAME_WIDTH;
canvas.height = GAME_HEIGHT;

// Remove the solid background from canvas to let space background show through
canvas.style.position = 'absolute';
canvas.style.left = '0';
canvas.style.top = '0';
canvas.style.backgroundColor = 'transparent';

// Background styles for the body
document.body.style.margin = '0';
document.body.style.padding = '0';
document.body.style.width = '100vw';
document.body.style.height = '100vh';
document.body.style.overflow = 'hidden';
document.body.style.backgroundImage = 'url(space.jpg)';
document.body.style.backgroundSize = 'cover';
document.body.style.backgroundPosition = 'center';
document.body.style.backgroundRepeat = 'no-repeat';

// Update resize handler
window.addEventListener('resize', () => {
  // No positioning updates needed
});

// streak related  constants
const STREAK_MESSAGES = ["Rampage!", "Oh yeah!", "Unstoppable!", "Savage!", "Sweet!", "Legendary!"];

const HOT_STREAK_WINDOW = 15000;            // measurement window for sterak msg  
const HOT_STREAK_MESSAGE_DURATION = 3000;  // 1 second in milliseconds
let currentKillCount = 0;
let previousKillCount = 0;
let lastStreakCheckTime = 0;
let showHotStreakMessage = false;
let hotStreakMessageTimer = 0;
let currentStreakMessage = "";
// streak related variables above

// ther monster constants
const MONSTER_SLALOM_SPEED = 170;      // Speed during slalom movement
const MONSTER_SLALOM_AMPLITUDE = 350;  // Increased from 200 to 350 for wider swings
const MONSTER_VERTICAL_SPEED = 60;     // Reduced from 100 to 60 for slower descent
const MONSTER_SLALOM_FIRE_RATE = 1800; // Fire rate during slalom mode (2 seconds)


const MONSTER_MISSILE_INTERVAL = 500; // Time between monster's missile shots (500ms)
let lastMonsterMissileTime = 0;

// other state variables at the top
let lifeRemovalAnimation = null;
let vaxGoneImage = new Image();
vaxGoneImage.src = 'vax_gone.svg';

// state variables
let   monster = null;
let   monsterDirection = 1;             // 1 for right, -1 for left
let   lastMonsterTime = 0;
const MONSTER_INTERVAL = 6000;          // seconds between monster appearances
const MONSTER2_INTERVAL = 10000;        //econds between monster appearances

const MONSTER_SPEED = 175;           // pixels per second
const MONSTER_WIDTH = 56;            //  
const MONSTER_HEIGHT = 56;           //  
const MONSTER_HIT_DURATION = 700;     // 0.7 seconds
let   monsterHit = false;
let   monsterImage = new Image();
let   monsterHitImage = new Image();
monsterImage.src = 'monster.svg';
monsterHitImage.src = 'monster_shot.svg';

// Kamikaze enemy settings
const KAMIKAZE_MIN_TIME = 6000;  // Min time between kamikaze launches
const KAMIKAZE_MAX_TIME = 11000; // Max time between kamikaze launches
const KAMIKAZE_SPEED = 170;      // Kamikaze movement speed (pixels per second)
const KAMIKAZE_FIRE_RATE = 900;  // Fire rate in milliseconds
const KAMIKAZE_AGGRESSIVE_TIME = 4000;         // Time between kamikazes when < 25 enemies
const KAMIKAZE_VERY_AGGRESSIVE_TIME = 2200;    // Time between kamikazes when < 10 enemies
const KAMIKAZE_AGGRESSIVE_THRESHOLD = 26;      // First threshold (25 enemies)
const KAMIKAZE_VERY_AGGRESSIVE_THRESHOLD = 11; // Second threshold in number of enemies

let lifeGrant = false;
const PLAYER_LIVES = 6;    // starting lives
let bonusGrants = 0;       // start with no bonus
const BONUS2LIVES = 6;     // every n bonuses, player gets one life
const BULLET_SPEED = 300;  // Player bullet speed (pixels per second)
const ENEMY_BULLET_SPEED = BULLET_SPEED / 3; // Enemy bullet speed (1/3 of player bullet speed)
const HIT_MESSAGE_DURATION = 900;            // How long to show "HIT!" message in milliseconds
const PLAYER_HIT_ANIMATION_DURATION = 750;   // Duration in milliseconds   (0.5 seconds)
const MIN_MISSILE_INTERVAL = 3200;           // 3 seconds
const MAX_MISSILE_INTERVAL = 7200;           // 
const MISSILE_SPEED = 170;                   // pixels per second
let shotCounter = 0;       // during rapid fire, only sound every 

let nextMissileTime = 0;
let homingMissiles = [];
let homingMissileHits = 0; // every 5th missile shot down we give bonus
let missileImage = new Image();
missileImage.src = 'missile.svg';
// monster-related constants
const MONSTER2_WIDTH = 56;
const MONSTER2_HEIGHT = 56;
const MONSTER2_SPEED = 220;  // Slightly faster than monster1
const MONSTER2_SPIRAL_RADIUS = 100;
const MONSTER2_SPIRAL_SPEED = 3;
const MONSTER2_VERTICAL_SPEED = 40;

// image declarations
let monster2Image = new Image();
monster2Image.src = 'monster2.svg';

// onster state variables
let monster2 = null;
let lastMonster2Time = 0;

const MONSTER2_DISAPPEAR_TIME = 8000; // 8 seconds disappearance time

const MONSTER2_MIN_RETURN_TIME = 5000; // 5 seconds minimum return time
const MONSTER2_MAX_RETURN_TIME = 9000; // 9 seconds maximum return time

// other monster2 constants
const MONSTER2_PATTERNS = {
    2: 'spiral',      // Level 2: Spiral pattern
    3: 'zigzag',      // Level 3: Horizontal zigzag while descending
    4: 'figure8',     // Level 4: Figure 8 pattern
    5: 'bounce',      // Level 5: Bounce off screen edges
    6: 'wave',        // Level 6: Sinusoidal wave pattern
    7: 'teleport',    // Level 7: Random teleportation
    8: 'chase',       // Level 8: Chase player with prediction
    9: 'random'       // Level 9+: Random quick movements
};


let player = {
  x: canvas.width / 2 - 37,
  y: canvas.height - 30,
  width: 48,
  height: 48,
  dx: 5,
  lives: PLAYER_LIVES,
  image: new Image(),
};

//let bonusgrants = 0; // tracks how many bonuses player got, every 7th lives++
let bullets = [];
let enemies = [];
let explosions = [];
let score = 0;
let enemyHitsToDestroy = 2; // how many times an enemy needs to be hit
let enemySpeed = 0.54;  // Decreased from 0.55
let enemyDirection = 1; // 1 for right, -1 for left
let gamePaused = false;
let lastFireTime = 0;
let gameOverFlag = false;
let victoryFlag = false;
let lastTime = 0;
const PLAYER_SPEED = 300;    // pixels per second
const ENEMY_SPEED = 50;      // pixels per second
const FIRE_RATE = 0.16;      // Time in seconds between shots (0.1 = 10 shots per second)
const ENEMY_FIRE_RATE = 0.72; // Time in seconds between enemy shots
let lastEnemyFireTime = 0;

let hitMessageTimer = 0;
let showHitMessage = false;

let playerHitTimer = 0;
let isPlayerHit = false;
let whilePlayerHit = false;
let playerNormalImage = new Image();
let playerExplosionImage = new Image();

playerNormalImage.src = "vax.svg";
playerExplosionImage.src = "player_explosion.svg";
player.image = playerNormalImage;

const KAMIKAZE_HITS_TO_DESTROY = 2;  // Number of hits needed to destroy a kamikaze


let kamikazeExplosionImage = new Image();
kamikazeExplosionImage.src = 'explode_kamikaze.svg';

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
  KeyB: false,
};

let wallImage = new Image();
wallImage.src = 'wall.svg';
let chunkImage = new Image();
chunkImage.src = 'chunk.png';

const WALL_HITS_FROM_BELOW = 3;  // hits needed for wall damage from player shots
const WALL_MAX_HITS_TOTAL = 11;  // total hits before wall disappears
const WALL_MAX_MISSILE_HITS = 4; // hits from missiles before wall disappears

// Original wall positions with evenly spaced walls
const INITIAL_WALLS = [
    {
        x: canvas.width * 1/5 - 29,  // First wall at 1/5
        y: canvas.height - 75,
        width: 58,
        height: 23,
        image: wallImage
    },
    {
        x: canvas.width * 2/5 - 29,  // Second wall at 2/5
        y: canvas.height - 75,
        width: 58,
        height: 23,
        image: wallImage
    },
    {
        x: canvas.width * 3/5 - 29,  // Third wall at 3/5
        y: canvas.height - 75,
        width: 58,
        height: 23,
        image: wallImage
    },
    {
        x: canvas.width * 4/5 - 29,  // Fourth wall at 4/5
        y: canvas.height - 75,
        width: 58,
        height: 23,
        image: wallImage
    }
];

let walls = INITIAL_WALLS.map(wall => ({
    ...wall,
    hitCount: 0,
    missileHits: 0
}));

// Initialize wallHits array for all walls
let wallHits = walls.map(() => []);

//counter for tracking explosions
let explosionCounter = 0;

// At the start of the game where other assets are loaded
let explosionImg = new Image();
explosionImg.src = 'explosion.svg';
let explosionAdditionalImg = new Image();
explosionAdditionalImg.src = 'explosion_additional.svg';

const BASE_FIRE_RATE = 0.190;           // Base time in seconds between shots
let currentFireRate = BASE_FIRE_RATE; // Current fire rate that can be modified

// for enemies  
const BASE_ENEMY_FIRE_RATE = 0.85;    // Base time in seconds between enemy shots
const ENEMY_FIRE_RATE_INCREASE = 0.10;// % increase per level
let currentEnemyFireRate = BASE_ENEMY_FIRE_RATE;

//  other initialization code
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
    //  bullet creation here
    if (!gamePausedt && Date.now() - lastFireTime > currentFireRate * 1000) {
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

  // touchstart and click events for better response
  touchFireLeft.addEventListener('touchstart', handleFireStart);
  touchFireLeft.addEventListener('touchend', handleFireEnd);
  touchFireRight.addEventListener('touchstart', handleFireStart);
  touchFireRight.addEventListener('touchend', handleFireEnd);
}

// handler to update tablet detection
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
  const enemyWidth = 43;  // Increased from 38
  const padding = 16;     // Increased from 14
  const minTotalWidth = (enemyWidth + padding) * minCols;

  // Calculate how many columns can fit in the current window width
  let cols = Math.floor((canvas.width - 60) / (enemyWidth + padding)); // 60 is total side padding
  cols = Math.max(minCols, Math.min(maxCols, cols)); // Clamp between min and max

  const enemyHeight = 43; // Increased from 38

  // Calculate the optimal starting position based on canvas height
  // Use a percentage of canvas height instead of fixed pixels
  const maxOffsetTop = 35; // minimum starting position
  const desiredOffsetTop = Math.min(canvas.height * 0.2, maxOffsetTop); // 20% of canvas height or 70px, whichever is smaller

  // Calculate the ideal gap between enemies and walls
  const idealGapToWalls = canvas.height * 0.3; // 30% of canvas height
  const wallY = walls[0]?.y || (canvas.height - 75); // fallback if no walls

  // Calculate where the bottom row of enemies should end
  const bottomRowY = wallY - idealGapToWalls;

  // Calculate total height needed for all rows
  const totalEnemyHeight = rows * (enemyHeight + padding);

  // Calculate final offsetTop to position enemies properly
  const offsetTop = Math.max(desiredOffsetTop, bottomRowY - totalEnemyHeight);

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
      bullets = bullets.filter(b => !b.isEnemyBullet); // remove enemy bullets
      bullets = bullets.filter(b => b.isEnemyBullet);  // remove player bullets
      whilePlayerHit = true; // flag player is in hit mode and during this time we don't allow bullets
      if (Date.now() - playerHitTimer > PLAYER_HIT_ANIMATION_DURATION) {
      isPlayerHit = false;
      player.image = playerNormalImage;
      player.width = 48;
      player.height = 48;
    } else {
      player.image = playerExplosionImage;
      player.width = 25 * 2;
      player.height = 25 * 2;
      // Only draw if image is loaded
      if (player.image.complete) {
        ctx.drawImage(
          player.image,
          player.x - (player.width / 2),
          player.y - (player.height / 2),
          48,
          48
        );
      }
      return;
    }
    whilePlayerHit = false; // flag player is not in hit mode
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
      if (bullet.isMonster2Bullet) {
        ctx.fillStyle = "#ff0000";  // Bright red for monster2 bullets
      } else {
        ctx.fillStyle = "#39ff14";  // Original neon green for other enemy bullets
      }
    } else {
      ctx.fillStyle = "white";    // Player bullets
    }
    ctx.fillRect(bullet.x, bullet.y, 3.4, 5.9);
  });
}

function drawWalls() {
  walls.forEach((wall, index) => {
    // Save the current context state
    ctx.save();
    
    // First draw the wall
    ctx.drawImage(wall.image, wall.x, wall.y, wall.width, wall.height);
    
    // Set up compositing to "cut out" the damage spots
    ctx.globalCompositeOperation = 'destination-out';
    
    // Draw the damage holes (they will create transparent areas)
    if (wallHits[index]) {
      wallHits[index].forEach(hit => {
        ctx.save();
        ctx.translate(wall.x + hit.x + 10, wall.y + hit.y);
        ctx.rotate(hit.rotation);
        
        // Create circular/oval holes instead of drawing chunk images
        ctx.beginPath();
        if (hit.fromEnemy) {
          // Elongated oval for enemy hits
          ctx.ellipse(0, 0, 10, 7, 0, 0, Math.PI * 2);
        } else {
          // Circular hole for player hits
          ctx.arc(0, 0, 10, 0, Math.PI * 2);
        }
        ctx.fill();
        
        ctx.restore();
      });
    }
    
    // Restore the original context state
    ctx.restore();
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
      if (!whilePlayerHit) bullet.y += ENEMY_BULLET_SPEED * deltaTime; // Slower enemy bullets
    } else {
      if (!whilePlayerHit) bullet.y -= BULLET_SPEED * deltaTime; // no bullets during player hit
    }
  });
  bullets = bullets.filter((bullet) =>
    bullet.y > 0 && bullet.y < canvas.height
  );
}

function moveEnemies(deltaTime) {
  // safety check for large deltaTime values
  if (deltaTime > 0.1) deltaTime = 0.1;  // Cap maximum deltaTime at 100ms

  // First check if there are any enemies at all
  if (enemies.length === 0) {
    if (!gameOverFlag) {
      gamePaused = true;
      victory();
    }
    return;
  }

  const currentEnemySpeed = (ENEMY_SPEED * enemySpeed) * deltaTime;
  let needsToMoveDown = false;

  // First check if any enemy needs to change direction
  enemies.forEach((enemy) => {
    // Skip any null or undefined enemies
    if (!enemy) return;

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
      // Skip any null or undefined enemies
      if (!enemy) return;

      // Only check active enemies
      if (enemy.hits < enemyHitsToDestroy) {
        enemy.y += moveDownAmount;
        // Check if there are any walls before checking position
        const wallY = walls.length > 0 ? walls[0].y - 20 : canvas.height * 0.90;

        if (enemy.y + enemy.height >= wallY) {
           
          gameOverFlag = true;
          gameOver();
        }
      }
    });
  }

  // Clean up any destroyed enemies
  enemies = enemies.filter(enemy => enemy && enemy.hits < enemyHitsToDestroy);

  if (enemies.length === 0 && !gameOverFlag) {
    gamePaused = true;
    victory();
  }
}

function detectCollisions() {
  // Check bullet collisions with walls
  bullets.forEach((bullet, bulletIndex) => {
    // kamikaze-bullet collision detection
    if (!bullet.isEnemyBullet) {
      kamikazeEnemies.forEach((kamikaze, kIndex) => {
        if (bullet.x < kamikaze.x + kamikaze.width &&
            bullet.x + 5 > kamikaze.x &&
            bullet.y < kamikaze.y + kamikaze.height &&
            bullet.y + 10 > kamikaze.y) {
          // Remove bullet
          bullets.splice(bulletIndex, 1);
          
          // Increment hit counter
          kamikaze.hits++;
          
          // Check if kamikaze is destroyed
          if (kamikaze.hits >= KAMIKAZE_HITS_TO_DESTROY) {
            // Create special kamikaze explosion
            explosions.push({
              x: kamikaze.x - 20, // Offset to center the explosion
              y: kamikaze.y - 20,
              frame: 0,
              img: kamikazeExplosionImage,
              width: kamikaze.width * 2.2,  // Make explosion bigger than the kamikaze
              height: kamikaze.height * 2,
              startTime: Date.now()
            });
            
            // kamikaze is killed, add points and increase kill count
            kamikazeEnemies.splice(kIndex, 1);
            score += 300;
            currentKillCount++; // kill count 
            
            // Play kamikaze explosion sound
            if (!isMuted) {
              kamikazeExplosionSound.currentTime = 0;
              kamikazeExplosionSound.play().catch(error => {
                console.log("Error playing kamikaze explosion sound:", error);
              });
            }
          }
          else {
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
          return;
        }
      });
    }

    // Existing wall collision check
    walls.forEach((wall, wallIndex) => {
      if (bullet.x < wall.x + wall.width &&
        bullet.x + 5 > wall.x &&
        bullet.y < wall.y + wall.height &&
        bullet.y + 10 > wall.y) {
        
        // chunk at bullet impact point with rotation info
        wallHits[wallIndex].push({
          x: bullet.x - wall.x - 10,
          y: bullet.isEnemyBullet ? 
              bullet.y - wall.y + 10 :   // Moved enemy bullet impact much lower
              bullet.y - wall.y,         // Player bullet position unchanged
          timeCreated: Date.now(),
          rotation: bullet.isEnemyBullet ? 0 : Math.PI
        });

        // Update wall damage
        if (!bullet.isEnemyBullet) {
          wall.hitCount++;
        } else {
          wall.missileHits++;
        }

        bullets.splice(bulletIndex, 1);
        return;
      }
    });
  });

  // kamikaze-player and kamikaze-wall collision detection
  if (!isPlayerHit) {
    kamikazeEnemies.forEach((kamikaze, kIndex) => {
      // Check wall collisions first
      let hitWall = false;
      walls.forEach((wall) => {
        if (kamikaze.x < wall.x + wall.width &&
            kamikaze.x + kamikaze.width > wall.x &&
            kamikaze.y < wall.y + wall.height &&
            kamikaze.y + kamikaze.height > wall.y) {
          createExplosion(kamikaze.x, kamikaze.y);
          kamikazeEnemies.splice(kIndex, 1);
          hitWall = true;
          return;
        }
      });

      // If didn't hit wall, check player collision
      if (!hitWall && 
          kamikaze.x < player.x + player.width &&
          kamikaze.x + kamikaze.width > player.x &&
          kamikaze.y < player.y + player.height &&
          kamikaze.y + kamikaze.height > player.y) {
        kamikazeEnemies.splice(kIndex, 1);
        handlePlayerHit();
        createExplosion(kamikaze.x, kamikaze.y);
      }
    });
  }

  // missile-player collision detection
  if (!isPlayerHit) {
    homingMissiles.forEach((missile, mIndex) => {
      const dx = (missile.x) - (player.x + player.width / 2);
      const dy = (missile.y) - (player.y + player.height / 2);
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < (player.width / 2 + missile.width / 4)) {
        homingMissiles.splice(mIndex, 1);
        handlePlayerHit();
        isPlayerHit = true;
        playerHitTimer = Date.now();
        player.image = playerExplosionImage;
        playerExplosionSound.currentTime = 0;
        playerExplosionSound.play();
        bullets = bullets.filter(b => !b.isEnemyBullet);
        homingMissiles = [];

        if (player.lives <= 0) {
          gameOverFlag = true;
          gameOverSound.currentTime = 0;
          gameOverSound.play();
        }
      }
    });
  }

  //  enemy bullet collision with walls
  bullets.forEach((bullet, bIndex) => {
    if (bullet.isEnemyBullet) {  // Check only enemy bullets
      walls.forEach((wall, wallIndex) => {
        if (bullet.x >= wall.x &&
          bullet.x <= wall.x + wall.width &&
          bullet.y >= wall.y &&
          bullet.y <= wall.y + wall.height) {

          // Remove the enemy bullet
          bullets.splice(bIndex, 1);

          // damage mark
          wallHits[wallIndex].push({
            x: bullet.x - wall.x,
            y: bullet.y - wall.y
          });

          // Count hits for this wall
          wall.hitCount = (wall.hitCount || 0) + 1;

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

        // homing missile is hit by player bullet ?
        if (distance < (missile.width / 2 + 5)) {
          homingMissileHits++;
          missileBoomSound.currentTime = 0;
          if (!isMuted) {
            missileBoomSound.play();
            setTimeout(() => {
              missileBoomSound.pause();
              missileBoomSound.currentTime = 0;
            }, 800);
          }
          if (homingMissileHits % 5 === 0) {
            score += 500; // bonus for every 4th missile shot down
            if (!isMuted) bonusSound.play(); // normal bonus sound

            /* every BONUS2LIVES (7 normally) bonus, lives++ but 
            not over PLAYER_LIVES max defined by programmer */
            bonusGrants++;
            if (bonusGrants >= BONUS2LIVES) { // normally 7
              player.lives++; // every nth  bonus grants player gets one life back!

              if (player.lives > PLAYER_LIVES) {
                player.lives = PLAYER_LIVES; // don't go over max
              } else {
                if (!isMuted) {
                  newLifeSound.volume = 1.0; // max volume
                  newLifeSound.play();
                }
                // Initialize the animation properties with debug logging
                //console.log('Starting life grant animation');
                lifeGrant = true;
                animations.lifeGrant = {
                  startTime: Date.now(),
                  startX: canvas.width / 2,
                  startY: canvas.height - 100  // Start a bit higher for better visibility
                };
              }
             
              bonusGrants = 0; // reset to zero again 
            }


            // Trigger bonus animation
            showBonusAnimation = true;
            bonusAnimationStart = Date.now();
          }

          score += 500;
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

/*          missileBoomSound.currentTime = 0;
          if (!isMuted) {
            missileBoomSound.play();
            setTimeout(() => {
              missileBoomSound.pause();
              missileBoomSound.currentTime = 0;
            }, 400);
            }*/            
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
          handlePlayerHit();

          // explosion animation and sound
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
            currentKillCount++; // kill count 
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
            // Skip bullet collision if monster is in slalom mode
            //if (monster.isSlaloming) {
            //    return;  // Skip collision check entirely
            //}

            // Rest of the existing monster collision code...
            const hasEnemyInPath = enemies.some(enemy =>
                bullet.x >= enemy.x &&
                bullet.x <= enemy.x + enemy.width &&
                bullet.y > enemy.y &&
                bullet.y < monster.y + monster.height
            );

            if (!hasEnemyInPath &&
                bullet.x < monster.x + monster.width &&
                bullet.x + 5 > monster.x &&
                bullet.y < monster.y + monster.height &&
                bullet.y + 10 > monster.y) {

                bullets.splice(bIndex, 1);
                monster.hit = true;
                monster.hitTime = Date.now();
                score += 500;

                // Restore walls to original positions
                walls = INITIAL_WALLS.map(wall => ({
                    ...wall,
                    hitCount: 0,
                    missileHits: 0
                }));
                
                // Reset wall hits array
                wallHits = walls.map(() => []);

                if (!isMuted) {
                    monsterDeadSound.currentTime = 0;
                    monsterDeadSound.play();
                }
            }
        }
    });
}

  // monster2 collision detection with player bullets
  if (monster2 && !monster2.isDisappeared) {
      bullets.forEach((bullet, bIndex) => {
          if (!bullet.isEnemyBullet) {
              // Skip bullet collision if monster2 is already hit
              if (monster2.hit) return;

              if (bullet.x < monster2.x + monster2.width &&
                  bullet.x + 5 > monster2.x &&
                  bullet.y < monster2.y + monster2.height &&
                  bullet.y + 10 > monster2.y) {
                  
                  bullets.splice(bIndex, 1);
                  monster2.hit = true;
                  monster2.hitTime = Date.now();
                  monster2.explosion = true; // explosion flag
                  score += 1500;  // Increased score for monster2

                  // Restore walls to original positions
                  walls = INITIAL_WALLS.map(wall => ({
                      ...wall,
                      hitCount: 0,
                      missileHits: 0
                  }));
                  
                  // Reset wall hits array
                  wallHits = walls.map(() => []);

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
  document.getElementById('score').innerHTML =
    `Score: ${score}<br>` +
    `Lives: ${player.lives}/${PLAYER_LIVES}<br>` +
    `Level: ${currentLevel}`;
  document.getElementById('version-info').textContent = VERSION;
}

function gameOver() {
  ctx.fillStyle = "white";
  ctx.font = "50px Arial";
  ctx.textAlign = "center";
  ctx.fillText("You lost! Game Over!", canvas.width / 2, canvas.height / 2);
  gamePaused = true;
}

// sound declarations at the top
let clearLevelSound = new Audio('clear-level-sfx.wav');

function victory() {
    currentLevel++;
    enemySpeed *= 1.33;  // Existing speed increase

    // Restore all walls to initial state
    walls = INITIAL_WALLS.map(wall => ({
        ...wall,
        hitCount: 0,
        missileHits: 0
    }));
    wallHits = walls.map(() => []);

    // Increase enemy fire rate by reducing the time between shots
    currentEnemyFireRate = BASE_ENEMY_FIRE_RATE /
        (1 + (ENEMY_FIRE_RATE_INCREASE * (currentLevel - 1)));

    score += 2500;  //points for completing the level
    
    // Clear all projectiles and enemies
    enemies = [];
    bullets = [];
    homingMissiles = [];  // Clear all missiles
    kamikazeEnemies = []; // Clear all kamikazes
    monster = null;       // Remove the monster

    // Play the level clear sound if not muted
    if (!isMuted) {
        clearLevelSound.currentTime = 0;
        clearLevelSound.play();
    }

    // Draw the level message
    ctx.save();
    ctx.fillStyle = "white";
    ctx.font = "48px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`Level ${currentLevel}`, canvas.width / 2, canvas.height / 2);
    ctx.restore();

    // Wait 1.5 seconds before starting the new level
    setTimeout(() => {
        createEnemies();
        gamePaused = false;
        victoryFlag = false;
        lastTime = 0;
        requestAnimationFrame(gameLoop);
    }, 1500);

    // Reset nextKamikazeTime for the new level
    nextKamikazeTime = performance.now() + 
        Math.random() * (KAMIKAZE_MAX_TIME - KAMIKAZE_MIN_TIME) + 
        KAMIKAZE_MIN_TIME;
}

function restartGame() {
  // Reset player to initial position and state
  player = {
    x: canvas.width / 2 - 37,  // Center player horizontally, offset by half width
    y: canvas.height - 30,     // Same vertical position as game start
    width: 48,                 // Original width
    height: 48,                // Original height
    dx: 5,
    lives: PLAYER_LIVES,
    image: playerNormalImage
  };

  // Reset other game variables
  score = 0;
  currentLevel = 1;
  gameOverFlag = false;
  isPlayerHit = false;
  whilePlayerHit = false;
  gamePaused = false;
  lastTime = 0;
  
  // Reset walls to initial state
  walls = INITIAL_WALLS.map(wall => ({
    ...wall,
    hitCount: 0,
    missileHits: 0
  }));
  wallHits = walls.map(() => []);
  
  // Reset enemies
  enemies = [];
  createEnemies();
  
  // Reset other game elements
  bullets = [];
  homingMissiles = [];
  monster = null;
  
  // Reset kamikaze-related variables
  kamikazeEnemies = [];
  nextKamikazeTime = performance.now() + 
      Math.random() * (KAMIKAZE_MAX_TIME - KAMIKAZE_MIN_TIME) + 
      KAMIKAZE_MIN_TIME;
  
  // Start the game loop
  requestAnimationFrame(gameLoop);
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
   // Only show PAUSED during regular gameplay
  if (gamePaused && !gameOverFlag && enemies.length > 0 && currentLevel === Math.floor(currentLevel)) { 
    ctx.save();
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "48px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("PAUSED", canvas.width / 2, canvas.height / 2);
    ctx.restore();
  }
}

function drawLevelMessage() {
  if (gamePaused && !gameOverFlag && enemies.length === 0) {  //Only show during level transition
    ctx.save();
    ctx.fillStyle = "white";
    ctx.font = "48px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`Level ${currentLevel}`, canvas.width / 2, canvas.height / 2);
    ctx.restore();
  }
}

function drawLives() {
  const LIFE_ICON_SIZE = 17;
  const PADDING = 3;
  const startX = canvas.width - LIFE_ICON_SIZE - PADDING;
  const startY = canvas.height - 20;

  ctx.save();
  ctx.fillStyle = '#39FF14';
  ctx.font = '15px Arial';
  ctx.textAlign = 'right';
  ctx.fillText('Lives', startX + PADDING, startY - 3);

  for (let i = 0; i < player.lives; i++) {
    ctx.drawImage(
      playerNormalImage,
      startX - (i * (LIFE_ICON_SIZE + PADDING)),
      startY,
      LIFE_ICON_SIZE,
      LIFE_ICON_SIZE
    );
  }

  // Draw removal animation if active
  if (lifeRemovalAnimation) {
    const now = Date.now();
    if (now - lifeRemovalAnimation.startTime < 1500) { // 1.5 seconds
      const x = startX - (lifeRemovalAnimation.position * (LIFE_ICON_SIZE + PADDING));
      ctx.drawImage(
        vaxGoneImage,
        x - LIFE_ICON_SIZE, // Bigger area
        startY - LIFE_ICON_SIZE,
        LIFE_ICON_SIZE * 2, // 3x size for emphasis
        LIFE_ICON_SIZE * 2
      );
    } else {
      lifeRemovalAnimation = null;
    }
  }

  ctx.restore();
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
  if (keys.Space && !isPlayerHit && !whilePlayerHit && !gamePaused &&
    Date.now() - lastFireTime > currentFireRate * 1000) {
    
    // Create the bullet
    bullets.push({
        x: player.x + player.width / 2 - 2.5,
        y: player.y,
        isEnemyBullet: false
    });
    lastFireTime = Date.now();
    
    // Increment shot counter
    shotCounter++;
    
    // Check if we're in rapid fire mode
    const isRapidFire = Date.now() - spaceKeyPressTime > MACHINE_GUN_THRESHOLD;
    
    // Play sound only on every second shot during rapid fire
    // or on every shot during normal firing
    if (!isRapidFire || shotCounter % 3 === 0) {
        playSoundWithCleanup(() => playerShotSound);
    }
}

  // Game logic
  if (!gamePaused && !gameOverFlag) {
    createMonster(currentTime);
    createMonster2(currentTime);  
    moveMonster(deltaTime);
    moveMonster2(deltaTime);        // for monster2
    updateKillStreak(currentTime);  // for hot streak msg 

    // kamikaze enemy creation
    if (currentTime >= nextKamikazeTime) {
      const enemy = getRandomEnemy();
      if (enemy) {
        // Remove the chosen enemy from the enemies array
        enemies = enemies.filter(e => e !== enemy);
        
        // Convert it to a kamikaze
        kamikazeEnemies.push({
          x: enemy.x,
          y: enemy.y,
          width: enemy.width,
          height: enemy.height,
          angle: 0,
          time: 0,
          hits: 0,  // hit counter
          lastFireTime: performance.now(),  // Initialize lastFireTime
          image: enemy.image
        });
        
        // Play kamikaze launch sound at max volume
        if (!isMuted) {
          kamikazeLaunchSound.currentTime = 0;
          kamikazeLaunchSound.play().catch(error => {
            console.log("Error playing kamikaze launch sound:", error);
          });
        }
        
        // Set next kamikaze time based on number of remaining enemies
        if (enemies.length < KAMIKAZE_VERY_AGGRESSIVE_THRESHOLD) {
          nextKamikazeTime = currentTime + KAMIKAZE_VERY_AGGRESSIVE_TIME;
        } else if (enemies.length < KAMIKAZE_AGGRESSIVE_THRESHOLD) {
          nextKamikazeTime = currentTime + KAMIKAZE_AGGRESSIVE_TIME;
        } else {
          nextKamikazeTime = currentTime +
              Math.random() * (KAMIKAZE_MAX_TIME - KAMIKAZE_MIN_TIME) +
              KAMIKAZE_MIN_TIME;
        }
      }
    }
  }

  // Draw everything
  drawPlayer();
  drawEnemies();
  drawKamikazeEnemies();
  drawMonster();
  drawMonster2(); 
  drawBullets();
  drawMissiles();
  drawMissileExplosions();
  drawWalls();
  drawExplosions();
  drawScore();
  drawHitMessage();
  drawMuteStatus();
  drawLevelMessage();
  drawLives();
  drawPauseMessage();
  drawLifeGrant();
  drawAIStatus();
  drawBonusAnimation();

  // Update game elements if not paused
  if (!gamePaused && !gameOverFlag) {
    updateAutoPlay();
    movePlayer(deltaTime);
    if (player.lives > 0) {
      moveBullets(deltaTime);
      moveEnemies(deltaTime);
      moveKamikazeEnemies(deltaTime);
      moveMissiles(deltaTime);
      handleEnemyShooting(currentTime);
      handleMissileLaunching(currentTime);
      detectCollisions();
    }
  }

  // Draw game over message if needed
  if (gameOverFlag) {
    ctx.fillStyle = "white";
    ctx.font = "60px Arial";
    ctx.textAlign = "center";
    ctx.fillText("GAME OVER!", canvas.width / 2, canvas.height / 2);
    ctx.fillStyle = "cyan";
    ctx.font = "35px Arial";
    ctx.textAlign = "center";
    //ctx.fillText("press R to restart the game", canvas.width / 2, canvas.height / 2 + 150);
  }

  // Draw hot streak message last so it appears on top of everything
  drawHotStreakMessage();

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
  //console.log("Key pressed:", e.code);  // Debug line to verify key detection

  if (e.code === 'KeyB') {
    const bossScreen = document.getElementById('boss-screen');
    const wasGamePaused = gamePaused;

    if (bossScreen.style.display === 'none' || bossScreen.style.display === '') {
      // Show boss screen
      bossScreen.style.display = 'block';
      gamePaused = true;
    } else {
      // Hide boss screen and resume game
      bossScreen.style.display = 'none';
      gamePaused = wasGamePaused;
      if (!gamePaused) {
        lastTime = 0;
        requestAnimationFrame(gameLoop);
      }
    }
  }
  if (e.code === "KeyP") {
    gamePaused = !gamePaused;
    if (!gamePaused) {
      lastTime = 0;
      requestAnimationFrame(gameLoop);
    }
  }
  if (e.code === "KeyR") {
    //restartGame();
  }
  if (e.code in keys) {
    keys[e.code] = true;
    if (e.code === "Space" && !keys.Space) {
      spaceKeyPressTime = Date.now();
    }
  }
  if (e.code === "KeyM") {
    isMuted = !isMuted;
    playerExplosionSound.muted = isMuted;
    gameOverSound.muted = isMuted;
  }
  if (e.code === "F1") {
    autoPlayEnabled = !autoPlayEnabled;
    // Reset movement keys when toggling to prevent stuck movement
    keys.ArrowLeft = false;
    keys.ArrowRight = false;
    keys.Space = false;
  }
  //if (e.code === "F10") {player.lives++}
  //if (e.code === "F9")  {player.lives--}
                                             
//  if (e.code === "F11") {
//    e.preventDefault(); // Prevent default F11 fullscreen behavior
    // Force move to next level
 //   currentLevel++; // Increment level
 //   enemies = []; // Clear enemies
//    bullets = []; // Clear bullets
 //   homingMissiles = []; // Clear missiles
 //   kamikazeEnemies = []; // Clear kamikazes
 //   monster = null; // Remove monster
 //   monster2 = null; // Remove monster2
    
 // Restore walls
//    walls = INITIAL_WALLS.map(wall => ({
//        ...wall,
//        hitCount: 0,
////        missileHits: 0
//    }));
//    wallHits = walls.map(() => []);
//    
//    // Create new enemies for next level
//    createEnemies();
 //   gamePaused = false;
 //   victoryFlag = false;
 //   lastTime = 0;
 //   requestAnimationFrame(gameLoop);
 // }
 //
 
 
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

let playerExplosionSound = new Audio('player_explosion.wav');
let startGameSound = new Audio('startgame.mp3');
let gameOverSound = new Audio('overgame.mp3');
let monsterDeadSound = new Audio('monster_dead.mp3');
let bonusSound = new Audio('bonus.mp3');   // for every 5th missile shot down 
let newLifeSound = new Audio('tadaa.mp3'); // new life granted!
let playerShotSound = new Audio('playershot3.mp3');
let machineGunSound = new Audio('mgun.mp3');
let spaceKeyPressTime = 0;
const MACHINE_GUN_THRESHOLD = 500;          // 0.5 seconds in milliseconds


let kamikazeExplosionSound = new Audio('explode_kamikaze.mp3');


let kamikazeLaunchSound = new Audio('launch_kamikaze.mp3');
kamikazeLaunchSound.volume = 1.0; // Set to max volume

let currentLevel = 1;

// get random enemy from top row
function getRandomTopRowEnemy() {
  const topY = Math.min(...enemies.map(e => e.y));
  const topRowEnemies = enemies.filter(e => e.y === topY);
  return topRowEnemies[Math.floor(Math.random() * topRowEnemies.length)];
}

// handle missile launching
function handleMissileLaunching(currentTime) {
  if (currentTime >= nextMissileTime) {
    const shooter = getRandomTopRowEnemy();
    if (shooter) {
      homingMissiles.push({
        x: shooter.x + shooter.width / 2,
        y: shooter.y + shooter.height,
        angle: 0,
        width: 57,  // Increased from 51
        height: 57, // Increased from 51
        time: 0
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

//  move missiles
function moveMissiles(deltaTime) {
  homingMissiles.forEach(missile => {
    missile.time += deltaTime;

    // Check if there are any walls left before checking position
    const wallRowY = walls.length > 0 ? walls[0].y - 50 : canvas.height * 0.85;

    // Check if missile is above wall row
    const isAboveWallRow = missile.y < wallRowY;

    if (isAboveWallRow) {
      // Calculate target direction
      const dx = player.x + player.width / 2 - missile.x;
      const dy = player.y + player.height / 2 - missile.y;

      // Calculate missile angle
      missile.angle = Math.atan2(dy, dx);
    }
    // else keep the last angle

    //  curved trajectory for rockets
    const curve = Math.sin(missile.time * 2) * 100;

    // Move missile
    missile.x += Math.cos(missile.angle) * MISSILE_SPEED * deltaTime;
    missile.y += Math.sin(missile.angle) * MISSILE_SPEED * deltaTime;
    missile.x += Math.cos(missile.angle + Math.PI / 2) * curve * deltaTime;
  });

  // Remove missiles that are off screen
  homingMissiles = homingMissiles.filter(m =>
    m.y < canvas.height && m.y > 0 && m.x > 0 && m.x < canvas.width
  );
}

// draw missiles
function drawMissiles() {
  homingMissiles.forEach(missile => {
    ctx.save();
    ctx.translate(missile.x, missile.y);
    ctx.rotate(missile.angle + Math.PI / 2);
    ctx.drawImage(
      missileImage,
      -missile.width / 2,
      -missile.height / 2,
      33,  // Increased from 29
      33   // Increased from 29
    );
    ctx.restore();
  });
}

// homing missile is hit 
let missileBoomSound = new Audio('explode_missile.mp3');

let missileExplosionImage = new Image();
missileExplosionImage.src = 'explode_missile.svg';

// missile explosions array with other state variables
let missileExplosions = [];

// draw missile explosions
function drawMissileExplosions() {
  missileExplosions = missileExplosions.filter(explosion => {
    const age = Date.now() - explosion.timeCreated;
    if (age < explosion.duration) {
      ctx.drawImage(
        missileExplosionImage,
        explosion.x - explosion.width / 2,
        explosion.y - explosion.height / 2,
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
          // console.log("Sound play aborted");
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
      bullets.push(createBullet(player.x + player.width / 2, player.y, false));
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



//let bonusImage = new Image();
//bonusImage.src = 'bonus.svg'; // for every 5th missile shot down 

// handle monster creation
function createMonster(currentTime) {
    if (!monster && currentTime - lastMonsterTime > MONSTER_INTERVAL) {
        monsterDirection = Math.random() < 0.5 ? 1 : -1;

        // Calculate starting position - start just off screen
        const startX = monsterDirection === 1 ? -MONSTER_WIDTH : canvas.width + MONSTER_WIDTH;
        const topEnemyRow = Math.min(...enemies.map(e => e.y)) - 50;

        // Check if we should enable slalom mode based on enemy count
        const shouldSlalom = enemies.length < ( KAMIKAZE_AGGRESSIVE_THRESHOLD -7) ;

        monster = {
            x: startX,
            y: shouldSlalom ? 0 : Math.max(topEnemyRow, MONSTER_HEIGHT) - 45,
            width: MONSTER_WIDTH,
            height: MONSTER_HEIGHT,
            hit: false,
            hitTime: 0,
            hasShot: false,
            slalomTime: 0,
            startY: 0,
            isSlaloming: shouldSlalom,
            lastFireTime: performance.now()  // for tracking missile firing
        };

        lastMonsterTime = currentTime;
    }
}

//  move monster
function moveMonster(deltaTime) {
    if (monster) {
        if (monster.hit) {
            if (Date.now() - monster.hitTime > MONSTER_HIT_DURATION) {
                monster = null;
                lastMonsterTime = performance.now();
            }
        } else {
            if (monster.isSlaloming) {
                // Slalom movement
                monster.slalomTime += deltaTime;
                
                // Calculate horizontal position using sine wave
                const centerX = canvas.width / 2;
                monster.x = centerX + Math.sin(monster.slalomTime * 1.2) * MONSTER_SLALOM_AMPLITUDE;
                
                // Move downward
                monster.y += MONSTER_VERTICAL_SPEED * deltaTime;
                
                // Fire single missile at MONSTER_SLALOM_FIRE_RATE intervals
                const currentTime = performance.now();
                if (currentTime - monster.lastFireTime >= MONSTER_SLALOM_FIRE_RATE) {
                    homingMissiles.push({
                        x: monster.x + monster.width / 2,
                        y: monster.y + monster.height,
                        angle: Math.PI / 2,
                        width: 44,
                        height: 44,
                        time: 0,
                        fromMonster: true
                    });
                    
                    playSoundWithCleanup(createMissileLaunchSound);
                    monster.lastFireTime = currentTime;
                }
                
                // Reset monster when it reaches just above the walls
                const wallY = walls.length > 0 ? walls[0].y : canvas.height * 0.85;
                const targetY = wallY - monster.height - 20;
                
                if (monster.y >= targetY) {
                    monster = null;
                    lastMonsterTime = performance.now();
                }
            } else {
                // Original horizontal movement
                monster.x += MONSTER_SPEED * monsterDirection * deltaTime;
                
                // Original missile firing logic with double missiles
                const isFullyOnScreen = monster.x >= 0 && monster.x + monster.width <= canvas.width;
                if (!monster.hasShot && isFullyOnScreen) {
                    // Double missiles from left and right positions
                    const missileOffsets = [-monster.width / 4, monster.width / 4];
                    missileOffsets.forEach(offset => {
                        homingMissiles.push({
                            x: monster.x + (monster.width / 2) + offset,
                            y: monster.y + monster.height,
                            angle: Math.PI / 2,
                            width: 44,
                            height: 44,
                            time: 0,
                            fromMonster: true
                        });
                    });
                    
                    playSoundWithCleanup(createMissileLaunchSound);
                    monster.hasShot = true;
                }
                
                // Original removal logic
                if ((monsterDirection === 1 && monster.x > canvas.width + MONSTER_WIDTH) ||
                    (monsterDirection === -1 && monster.x < -MONSTER_WIDTH)) {
                    monster = null;
                    lastMonsterTime = performance.now();
                }
            }
        }
    }
}

// function to draw monster
function drawMonster() {
  if (monster) {
    const image = monster.hit ? monsterHitImage : monsterImage;
    if (image.complete) {
      ctx.drawImage(image, monster.x, monster.y, monster.width, monster.height);
    }
  }
}

// sound for monster hit
function createMonsterHitSound() {
  const sound = new Audio('monster_hit.mp3');
  sound.volume = 1.0;
  return sound;
}



// Modify where player gets hit (in detectCollisions or similar)
function handlePlayerHit() {
    player.lives--;
    showHitMessage = true;
    hitMessageTimer = Date.now();

    // life removal animation
    lifeRemovalAnimation = {
        startTime: Date.now(),
        position: player.lives
    };

    isPlayerHit = true;
    playerHitTimer = Date.now();
    player.image = playerExplosionImage;
    playerExplosionSound.currentTime = 0;
    playerExplosionSound.play();

    // Clear all enemy bullets, missiles, and kamikazes
    bullets = bullets.filter(b => !b.isEnemyBullet);
    homingMissiles = [];  // Clear all missiles in flight
    kamikazeEnemies = []; // Clear all kamikazes in flight

    if (player.lives <= 0) {
        gameOverFlag = true;
        gameOverSound.currentTime = 0;
        gameOverSound.play();
    }
}

// near other state variables
let bonusImage = new Image();
bonusImage.src = 'bonus.svg';
let showBonusAnimation = false;
let bonusAnimationStart = 0;
const BONUS_ANIMATION_DURATION = 1000; // 1 second to show bonus

// draw bonus animation
function drawBonusAnimation() {
  if (showBonusAnimation) {
    const age = Date.now() - bonusAnimationStart;
    if (age < BONUS_ANIMATION_DURATION) {
      if (bonusImage.complete) {
        ctx.drawImage(
          bonusImage,
          player.x + player.width + 10, // Position next to player
          player.y,                     // At player's height
          25,  // width
          25   // height
        );
      }
    } else {
      showBonusAnimation = false;
    }
  }
}

// these constants near the top with other constants
const LIFE_GRANT_ANIMATION = {
  DURATION: 3500,  // Changed from 1500 to 3000 (3 seconds)
  ICON_SIZE: 100
};

// this with other state variables
let animations = {
  lifeGrant: {
    startTime: 0,
    startX: 0,
    startY: 0
  }
};

// near the top with other image loads
let lifeImage = new Image();
lifeImage.src = 'life.svg';

function drawLifeGrant() {
  if (lifeGrant) {
    const elapsed = Date.now() - animations.lifeGrant.startTime;
    const progress = Math.min(elapsed / LIFE_GRANT_ANIMATION.DURATION, 1);

    if (progress < 1) {
      const easeOut = 1 - Math.pow(1 - progress, 3); // Cubic ease-out

      ctx.save();
      ctx.globalAlpha = 1 - easeOut;

      // Reduced the movement multiplier from 0.6 to 0.3 for slower upward movement
      const currentY = animations.lifeGrant.startY - (canvas.height * 0.3 * easeOut);

      if (lifeImage.complete) {
        ctx.drawImage(
          lifeImage,
          animations.lifeGrant.startX - LIFE_GRANT_ANIMATION.ICON_SIZE / 2,
          currentY - LIFE_GRANT_ANIMATION.ICON_SIZE / 2,
          LIFE_GRANT_ANIMATION.ICON_SIZE - 5,
          LIFE_GRANT_ANIMATION.ICON_SIZE - 5
        );
      } else {
        // Fallback if image not loaded
        ctx.fillStyle = 'green';
        ctx.fillRect(
          animations.lifeGrant.startX - LIFE_GRANT_ANIMATION.ICON_SIZE / 2,
          currentY - LIFE_GRANT_ANIMATION.ICON_SIZE / 2,
          LIFE_GRANT_ANIMATION.ICON_SIZE,
          LIFE_GRANT_ANIMATION.ICON_SIZE
        );
      }

      ctx.restore();
    } else {
      lifeGrant = false;
    }
  }
}

// Make sure lifeImage is loaded properly
lifeImage.onload = () => {
  //console.log('Life image loaded successfully');
};

lifeImage.onerror = () => {
  //console.error('Failed to load life.svg');
};

// is AI enabled?
let autoPlayEnabled = false;

// for AI logic
function updateAutoPlay() {
  if (!autoPlayEnabled || gamePaused || gameOverFlag) return;

  const threats = [];
  const playerCenter = player.x + player.width / 2;

  // Check for both falling bullets and horizontally nearby bullets for AI mode
  function isUnderBullet(newX) {
    return bullets.some(bullet => {
      if (!bullet.isEnemyBullet) return false;

      // Check falling bullets above
      if (bullet.y < player.y) {
        const timeToImpact = (player.y - bullet.y) / ENEMY_BULLET_SPEED;
        if (timeToImpact > 0 && timeToImpact < 0.3) {
          if (!willBulletHitWall(bullet, timeToImpact)) {
            return Math.abs(bullet.x - newX) < player.width * 0.8;
          }
        }
      }

      // Check bullets at similar height
      const heightDiff = Math.abs(bullet.y - player.y);
      if (heightDiff < player.height * 1.5) {
        // If moving right, check bullets to the right
        if (newX > playerCenter) {
          return bullet.x > playerCenter && bullet.x < newX + player.width;
        }
        // If moving left, check bullets to the left
        else if (newX < playerCenter) {
          return bullet.x < playerCenter && bullet.x > newX - player.width;
        }
      }

      return false;
    });
  }

  // Will bullet hit wall?
  function willBulletHitWall(bullet, timeToImpact) {
    const futureY = bullet.y + (ENEMY_BULLET_SPEED * timeToImpact);
    return walls.some(wall => {
      if (bullet.x >= wall.x && bullet.x <= wall.x + wall.width) {
        return bullet.y < wall.y && futureY >= wall.y;
      }
      return false;
    });
  }

  // Collect threats
  bullets.forEach(bullet => {
    if (bullet.isEnemyBullet) {
      const timeToImpact = (player.y - bullet.y) / ENEMY_BULLET_SPEED;
      if (timeToImpact > 0 && timeToImpact < 1.0 && !willBulletHitWall(bullet, timeToImpact)) {
        threats.push({
          x: bullet.x,
          y: bullet.y,
          type: 'bullet',
          timeToImpact: timeToImpact,
          priority: timeToImpact < 0.5 ? 3 : 2
        });
      }
    }
  });

  // Rest of threat collection remains the same
  homingMissiles.forEach(missile => {
    const dx = playerCenter - missile.x;
    const dy = player.y - missile.y;
    const distanceToPlayer = Math.sqrt(dx * dx + dy * dy);

    if (distanceToPlayer < 200) {
      threats.push({
        x: missile.x,
        y: missile.y,
        type: 'missile',
        priority: 3
      });
    }
  });

  const bottomRowY = Math.max(...enemies.map(e => e.y));
  enemies.forEach(enemy => {
    const isBottomRow = Math.abs(enemy.y - bottomRowY) < 20;
    if (isBottomRow) {
      threats.push({
        x: enemy.x + enemy.width / 2,
        y: enemy.y,
        type: 'enemy',
        priority: 1
      });
    }
  });

  if (monster && !monster.hit) {
    threats.push({
      x: monster.x + monster.width / 2,
      y: monster.y + monster.height / 2,
      type: 'monster',
      priority: 4
    });
  }

  threats.sort((a, b) => b.priority - a.priority);

  // Reset controls
  keys.ArrowLeft = false;
  keys.ArrowRight = false;
  keys.Space = false;

  if (threats.length > 0) {
    const dangerousThreats = threats.filter(t =>
      (t.type === 'bullet' && t.timeToImpact < 0.5 && Math.abs(t.x - playerCenter) < 50) ||
      (t.type === 'missile' && Math.abs(t.x - playerCenter) < 60)
    );

    if (dangerousThreats.length > 0) {
      // Dodge mode
      const threat = dangerousThreats[0];
      const moveLeft = threat.x > playerCenter;
      const newPos = moveLeft ?
        playerCenter - PLAYER_SPEED / 30 :
        playerCenter + PLAYER_SPEED / 30;

      // Check movement safety
      const movementSafe = !isUnderBullet(newPos);

      if (movementSafe) {
        if (moveLeft && player.x > 0) {
          keys.ArrowLeft = true;
        } else if (!moveLeft && player.x < canvas.width - player.width) {
          keys.ArrowRight = true;
        }
      }
    }
  }
}

function drawAIStatus() {
  if (autoPlayEnabled) {
    ctx.save();
    ctx.fillStyle = '#39FF14'; // Bright green color
    ctx.font = '19px Arial';
    ctx.textAlign = 'right';
    ctx.fillText('AI ACTIVE', canvas.width - 10, 20);
    ctx.restore();
  }
}


let nextKamikazeTime = performance.now() + 
    Math.random() * (KAMIKAZE_MAX_TIME - KAMIKAZE_MIN_TIME) + 
    KAMIKAZE_MIN_TIME;
let kamikazeEnemies = [];

// createEnemies function
function getRandomEnemy() {
    if (enemies.length === 0) return null;
    return enemies[Math.floor(Math.random() * enemies.length)];
}

function moveKamikazeEnemies(deltaTime) {
    kamikazeEnemies.forEach((kamikaze, index) => {
        kamikaze.time += deltaTime;
        
        // Handle shooting
        const currentTime = performance.now();
        if (currentTime - kamikaze.lastFireTime >= KAMIKAZE_FIRE_RATE) {
            bullets.push({
                x: kamikaze.x + kamikaze.width / 2,
                y: kamikaze.y + kamikaze.height,
                isEnemyBullet: true
            });
            kamikaze.lastFireTime = currentTime;
        }
        
        // Check for collision with player first
        if (kamikaze.x < player.x + player.width &&
            kamikaze.x + kamikaze.width > player.x &&
            kamikaze.y < player.y + player.height &&
            kamikaze.y + kamikaze.height > player.y) {
            handlePlayerHit();
            createExplosion(kamikaze.x, kamikaze.y);
            kamikazeEnemies.splice(index, 1);
            return;
        }
        
        // Check if kamikaze has reached player's height - remove it if so
        if (kamikaze.y >= player.y) {
            kamikazeEnemies.splice(index, 1);
            return;
        }
        
        // Calculate target direction
        const targetDx = player.x + player.width/2 - kamikaze.x;
        const targetDy = player.y + player.height/2 - kamikaze.y;
        
        // Calculate angle
        kamikaze.angle = Math.atan2(targetDy, targetDx);
        
        // curved trajectory like missiles
        const curve = Math.sin(kamikaze.time * 2) * 100;
        
        // Move kamikaze enemy
        kamikaze.x += Math.cos(kamikaze.angle) * KAMIKAZE_SPEED * deltaTime;
        kamikaze.y += Math.sin(kamikaze.angle) * KAMIKAZE_SPEED * deltaTime;
        kamikaze.x += Math.cos(kamikaze.angle + Math.PI/2) * curve * deltaTime;
        
        // Check wall collisions
        walls.forEach((wall) => {
            if (wall.hitCount < WALL_MAX_HITS_TOTAL && wall.missileHits < WALL_MAX_MISSILE_HITS) {
                if (kamikaze.x >= wall.x &&
                    kamikaze.x <= wall.x + wall.width &&
                    kamikaze.y >= wall.y &&
                    kamikaze.y <= wall.y + wall.height) {
                    createExplosion(kamikaze.x, kamikaze.y);
                    kamikazeEnemies.splice(index, 1);
                    wall.missileHits++;
                }
            }
        });
    });
    
    // Remove kamikaze enemies that are off screen horizontally
    kamikazeEnemies = kamikazeEnemies.filter(k =>
        k.x > 0 && k.x < canvas.width
    );
}

function drawKamikazeEnemies() {
    kamikazeEnemies.forEach(kamikaze => {
        ctx.save();
        ctx.translate(kamikaze.x + kamikaze.width / 2, kamikaze.y + kamikaze.height / 2);
        ctx.rotate(kamikaze.angle + Math.PI / 2);
        ctx.drawImage(
            kamikaze.image,
            -kamikaze.width / 2,
            -kamikaze.height / 2,
            kamikaze.width,
            kamikaze.height
        );
        ctx.restore();
    });
}


// check and update kill counts
function updateKillStreak(currentTime) {
    if (currentTime - lastStreakCheckTime >= HOT_STREAK_WINDOW) {
        if (currentKillCount > previousKillCount && previousKillCount > 0) {
            showHotStreakMessage = true;
            hotStreakMessageTimer = currentTime;
            // Select the random message here, when the streak is triggered
            currentStreakMessage = STREAK_MESSAGES[Math.floor(Math.random() * STREAK_MESSAGES.length)];
        }
        
        previousKillCount = currentKillCount;
        currentKillCount = 0;
        lastStreakCheckTime = currentTime;
    }
}


function drawHotStreakMessage() {
    if (showHotStreakMessage) {
        const currentTime = performance.now();
        if (currentTime - hotStreakMessageTimer < HOT_STREAK_MESSAGE_DURATION) {
            ctx.save();
            ctx.fillStyle = "white";
            ctx.font = "bold 17px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "left";
            
            // Use the pre-selected message
            ctx.fillText(currentStreakMessage, 12, canvas.height - 9);
            
            ctx.restore();
        } else {
            showHotStreakMessage = false;
        }
    }
}



// function to check if monster is in slalom mode
function isMonsterInSlalom() {
    return monster && monster.isSlaloming;
}

// Modify createMonster2 function to check for monster slalom
function createMonster2(currentTime) {
    // Only create monster2 in level 2 or higher
    if (currentLevel < 2) return;

    // Don't create monster2 if monster is in slalom mode
    if (isMonsterInSlalom()) {
        // Reset the timer to delay monster2 appearance
        lastMonster2Time = currentTime;
        return;
    }

    // Additional check to ensure monster has been gone for a while after slalom
    const monsterJustFinishedSlalom = !monster && 
        (currentTime - lastMonsterTime < MONSTER2_INTERVAL / 2);
    
    if (monsterJustFinishedSlalom) {
        return; // Wait longer before spawning monster2
    }

    // Original monster2 creation logic
    if (!monster2 && currentTime - lastMonster2Time > MONSTER2_INTERVAL) {
        monster2 = {
            x: canvas.width / 2,
            y: -MONSTER2_HEIGHT,
            width: MONSTER2_WIDTH,
            height: MONSTER2_HEIGHT,
            spiralAngle: 0,
            centerX: canvas.width / 2,
            hit: false,
            hitTime: 0,
            lastFireTime: currentTime
        };
        lastMonster2Time = currentTime;
    }
}

// Modify moveMonster2 function to restore bullet shooting
function moveMonster2(deltaTime) {
    if (!monster2) return;

    if (monster2.hit) {
        if (Date.now() - monster2.hitTime <= MONSTER_HIT_DURATION) {
            // Show explosion during hit animation
            if (monster2.explosion) {
                ctx.drawImage(
                    explosionAdditionalImg,
                    monster2.x - monster2.width/2,
                    monster2.y - monster2.height/2,
                    monster2.width * 2,
                    monster2.height * 2
                );
            }
        } else {
            // After hit animation, make monster disappear
            monster2.isDisappeared = true;
            monster2.disappearTime = Date.now();
            monster2.returnDelay = MONSTER2_MIN_RETURN_TIME + 
                Math.random() * (MONSTER2_MAX_RETURN_TIME - MONSTER2_MIN_RETURN_TIME);
            monster2.hit = false;
        }
        return;
    }

    // Check if monster2 is in disappeared state
    if (monster2.isDisappeared) {
        if (Date.now() - monster2.disappearTime > monster2.returnDelay) {
            // Reset monster2 position when returning
            monster2.isDisappeared = false;
            monster2.x = canvas.width / 2;
            monster2.y = -MONSTER2_HEIGHT;
            monster2.spiralAngle = 0;
            monster2.centerX = canvas.width / 2;
        } else {
            return; // Skip movement and shooting while disappeared
        }
    }

    // Get movement pattern based on level
    const pattern = currentLevel <= 6 ? MONSTER2_PATTERNS[currentLevel] : 'random';
    
    // Base vertical movement
    monster2.y += MONSTER2_VERTICAL_SPEED * deltaTime;

    // Apply pattern-specific movement
    switch (pattern) {
        case 'spiral':
            // Current spiral pattern
            monster2.spiralAngle += MONSTER2_SPIRAL_SPEED * deltaTime;
            const radius = MONSTER2_SPIRAL_RADIUS * Math.min(1, monster2.y / 200);
            monster2.x = monster2.centerX + Math.cos(monster2.spiralAngle) * radius;
            break;

        case 'zigzag':
            // Enhanced zigzag pattern with slower vertical movement
            if (!monster2.zigzagDir) {
                monster2.zigzagDir = 1;
                monster2.zigzagAmplitude = canvas.width * 0.4; // 40% of screen width
                monster2.zigzagBaseY = monster2.y;
                monster2.zigzagPhase = 0;
            }
            
            // Update phase
            monster2.zigzagPhase += deltaTime * 1.5;
            
            // Calculate horizontal position using zigzag pattern
            monster2.x = canvas.width/2 + Math.sin(monster2.zigzagPhase) * monster2.zigzagAmplitude;
            
            // Slow down vertical movement to 1/3 of normal speed
            monster2.y += MONSTER2_VERTICAL_SPEED * deltaTime * 0.33;
            
            // slight vertical oscillation for more interesting movement
            monster2.y += Math.sin(monster2.zigzagPhase * 2) * deltaTime * 15;
            
            // Ensure monster stays within horizontal bounds
            if (monster2.x < 0) monster2.x = 0;
            if (monster2.x > canvas.width - monster2.width) monster2.x = canvas.width - monster2.width;
            break;

        case 'figure8':
            // Figure 8 pattern
            monster2.spiralAngle += MONSTER2_SPIRAL_SPEED * deltaTime;
            monster2.x = monster2.centerX + Math.cos(monster2.spiralAngle) * MONSTER2_SPIRAL_RADIUS;
            monster2.y += Math.sin(2 * monster2.spiralAngle) * deltaTime * 30;
            break;

        case 'bounce':
            // Enhanced bouncing pattern with random direction changes
            if (!monster2.dx) {
                // Initialize with random directions
                monster2.dx = (Math.random() > 0.5 ? 1 : -1) * MONSTER2_SPEED;
                monster2.dy = (Math.random() > 0.5 ? 1 : -1) * MONSTER2_SPEED * 0.7;
                monster2.lastDirectionChange = Date.now();
                monster2.directionChangeInterval = 1500 + Math.random() * 1500; // 1.5-3 seconds
            }
            
            // Move according to current direction
            monster2.x += monster2.dx * deltaTime;
            monster2.y += monster2.dy * deltaTime;
            
            // Bounce off edges
            if (monster2.x <= 0) {
                monster2.x = 0;
                monster2.dx = Math.abs(monster2.dx);
            } else if (monster2.x >= canvas.width - monster2.width) {
                monster2.x = canvas.width - monster2.width;
                monster2.dx = -Math.abs(monster2.dx);
            }
            
            if (monster2.y <= 0) {
                monster2.y = 0;
                monster2.dy = Math.abs(monster2.dy);
            } else if (monster2.y >= canvas.height * 0.7) {
                // Don't let it go too far down
                monster2.y = canvas.height * 0.7;
                monster2.dy = -Math.abs(monster2.dy);
            }
            
            // Randomly change direction occasionally
            if (Date.now() - monster2.lastDirectionChange > monster2.directionChangeInterval) {
                // 30% chance to change x direction, 30% chance to change y direction
                if (Math.random() < 0.3) {
                    monster2.dx *= -1;
                }
                if (Math.random() < 0.3) {
                    monster2.dy *= -1;
                }
                
                // 20% chance to completely randomize direction
                if (Math.random() < 0.2) {
                    const angle = Math.random() * Math.PI * 2;
                    const speed = MONSTER2_SPEED * (0.8 + Math.random() * 0.4); //80-120% of base spd
                    monster2.dx = Math.cos(angle) * speed;
                    monster2.dy = Math.sin(angle) * speed * 0.7; // Slower vertical movement
                }
                
                // Set next direction change time
                monster2.lastDirectionChange = Date.now();
                monster2.directionChangeInterval = 1500 + Math.random() * 1500;
            }
            break;

        case 'wave':
            // Sinusoidal wave pattern
            if (!monster2.waveStartX) monster2.waveStartX = monster2.x;
            monster2.x = monster2.waveStartX + 
                Math.sin(monster2.y / 50) * (canvas.width / 4);
            break;

        case 'teleport':
            // Teleportation pattern
            if (!monster2.nextTeleportTime || Date.now() > monster2.nextTeleportTime) {
                // Store previous position for fade effect
                monster2.prevX = monster2.x;
                monster2.prevY = monster2.y;
                monster2.fadeStart = Date.now();
                
                // New random position in top 2/3 of screen
                monster2.x = Math.random() * (canvas.width - monster2.width);
                monster2.y = Math.random() * (canvas.height * 0.66);
                
                // Set next teleport time
                monster2.nextTeleportTime = Date.now() + 2000;
            }
            break;

        case 'chase':
            // Predictive chase pattern
            const predictedX = player.x + 
                (keys.ArrowRight ? 100 : keys.ArrowLeft ? -100 : 0);
            
            // Calculate direction to predicted position
            const chaseDx = predictedX - monster2.x;
            const chaseDy = player.y - monster2.y - 200; // Stay above player
            
            // Normalize and apply speed
            const chaseDistance = Math.sqrt(chaseDx * chaseDx + chaseDy * chaseDy);
            if (chaseDistance > 1) {
                monster2.x += (chaseDx / chaseDistance) * MONSTER2_SPEED * 1.2 * deltaTime;
                monster2.y += (chaseDy / chaseDistance) * MONSTER2_SPEED * 0.7 * deltaTime;
            }
            break;

        case 'random':
            // Random quick movements
            if (!monster2.nextMoveTime || Date.now() > monster2.nextMoveTime) {
                monster2.targetX = Math.random() * (canvas.width - monster2.width);
                monster2.targetY = Math.min(
                    Math.random() * canvas.height * 0.5,
                    monster2.y + 100
                );
                monster2.nextMoveTime = Date.now() + 1000;
            }
            
            // Move towards target
            const randomDx = monster2.targetX - monster2.x;
            const randomDy = monster2.targetY - monster2.y;
            const randomDistance = Math.sqrt(randomDx * randomDx + randomDy * randomDy);
            if (randomDistance > 1) {
                monster2.x += (randomDx / randomDistance) * MONSTER2_SPEED * deltaTime;
                monster2.y += (randomDy / randomDistance) * MONSTER2_SPEED * deltaTime;
            }
            break;
    }

    // bullet firing code 
    const currentTime = performance.now();
    if (currentTime - monster2.lastFireTime >= 2800) {
        // Fire 3 spread bullets in fixed directions
        for (let i = -1; i <= 1; i++) {
            const spreadAngle = Math.PI/2 + (i * Math.PI/8); // Spread around downward direction
            bullets.push({
                x: monster2.x + monster2.width/2,
                y: monster2.y + monster2.height,
                dx: Math.cos(spreadAngle) * ENEMY_BULLET_SPEED * 1.2,
                dy: Math.sin(spreadAngle) * ENEMY_BULLET_SPEED * 1.2,
                isEnemyBullet: true,
                isMonster2Bullet: true
            });
        }
        
        playSoundWithCleanup(createMissileLaunchSound);
        monster2.lastFireTime = currentTime;
    }

    // Remove if off screen
    if (monster2.y > canvas.height + MONSTER2_HEIGHT) {
        monster2 = null;
        lastMonster2Time = performance.now();
    }
}

// drawMonster2 function to handle teleport fade effect
function drawMonster2() {
    if (monster2 && monster2Image.complete && !monster2.isDisappeared) {
        // Draw fade effect for teleport pattern
        if (monster2.fadeStart && Date.now() - monster2.fadeStart < 200) {
            const alpha = 1 - ((Date.now() - monster2.fadeStart) / 200);
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.drawImage(
                monster2Image,
                monster2.prevX,
                monster2.prevY,
                monster2.width,
                monster2.height
            );
            ctx.restore();
        }

        // Draw current monster position
        ctx.save();
        ctx.translate(monster2.x + monster2.width/2, monster2.y + monster2.height/2);
        ctx.rotate(monster2.spiralAngle);
        ctx.drawImage(
            monster2Image,
            -monster2.width/2,
            -monster2.height/2,
            monster2.width,
            monster2.height
        );
        ctx.restore();
    }
}

// Modify moveBullets to remove tracking for monster2 bullets
function moveBullets(deltaTime) {
    bullets.forEach((bullet) => {
        if (bullet.isEnemyBullet) {
            if (bullet.dx !== undefined && bullet.dy !== undefined) {
                // For bullets with directional movement (monster2)
                bullet.x += bullet.dx * deltaTime;
                bullet.y += bullet.dy * deltaTime;
            } else {
                // Regular enemy bullets
                bullet.y += ENEMY_BULLET_SPEED * deltaTime;
            }
        } else {
            if (!whilePlayerHit) bullet.y -= BULLET_SPEED * deltaTime;
        }
    });
    
    bullets = bullets.filter((bullet) => 
        bullet.y > 0 && bullet.y < canvas.height && 
        bullet.x > 0 && bullet.x < canvas.width
    );
}
