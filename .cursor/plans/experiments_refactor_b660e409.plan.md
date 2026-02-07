---
name: Experiments Refactor
overview: Реорганізація структури проєкту з вкладеною системою експериментів, гнучкими параметрами POMDP, та окремими режимами для статичних і target агентів
todos:
  - id: create_experiment_py
    content: Створити src/experiment.py з ExperimentConfig, ExperimentDirectory, та функціями управління експериментами
    status: completed
  - id: create_scenarios_py
    content: Створити src/scenarios.py з PREDEFINED_SCENARIOS словником
    status: completed
  - id: create_static_script
    content: Створити run_static_agents.py для окремого запуску статичних агентів
    status: completed
  - id: update_utils
    content: "Оновити src/utils.py: видалити get_timestamped_results_dir(), додати consistent naming для всіх функцій"
    status: completed
  - id: update_train
    content: Оновити src/train.py для роботи з ExperimentDirectory та збереження в правильні папки
    status: completed
  - id: update_evaluation
    content: Оновити src/evaluation.py для прийняття exp_dir параметру та використання правильних шляхів
    status: completed
  - id: rewrite_main
    content: Переписати main.py з typer, додати --scenario, --load-experiment, dynamic POMDP flags, інтеграція з ExperimentDirectory
    status: completed
  - id: test_scenarios
    content: "Протестувати всі режими: predefined scenarios, custom params, load experiment, static agents script"
    status: completed
isProject: false
---

# План рефакторингу системи експериментів

## Мета

Створити чисту, масштабовану систему для управління експериментами з автоматичним збереженням результатів, конфігурацій, ваг і логів у структуровані папки.

## Нова структура директорій

### Вкладена організація (обрано користувачем)

```
experiments/
  mdp/
    2026-02-05_14-30-00/
      config.json          # Повна конфігурація (DefaultConfig + CLI параметри)
      summary.json         # Ключові метрики (peak infected, total rewards тощо)
      weights/
        ppo_baseline.zip   # RL агент 1
        ppo_framestack.zip # RL агент 2  
        ppo_recurrent.zip  # RL агент 3
      plots/
        comparison_all_agents.png    # Загальний графік з усіма агентами
        random_agent_seir.png         # Індивідуальні криві SEIR
        threshold_agent_seir.png
        ppo_baseline_seir.png
        ppo_baseline_learning.png     # Криві навчання (episodestimesteps)
      logs/
        random_agent.txt              # Детальні логи дій
        threshold_agent.txt
        ppo_baseline.txt
        tensorboard/                  # TensorBoard логи
          PPO_1/
            events.out.tfevents...
    2026-02-05_16-45-12/              # Інший запуск того ж експерименту
      ...
  no_exposed/
    2026-02-06_10-00-00/
      config.json
      ...
  custom/
    no_exposed_delay5_noise0.1_2026-02-06_12-00-00/
      config.json
      ...
```

### Переваги структури

- **Групування по експериментах**: легко знайти всі запуски одного сценарію
- **Історія запусків**: збережено всі спроби з timestamp
- **Самодокументація**: `config.json` містить усі параметри запуску
- **Легке завантаження**: можна знайти останній запуск експерименту для перезавантаження ваг

## Нові/змінені файли

### 1. `src/experiment.py` (новий файл)

**Призначення**: Управління експериментами, створення директорій, збереження конфігурацій.

**Ключові класи**:

```python
@dataclass
class ExperimentConfig:
    """Повна конфігурація експерименту (config + POMDP parameters)"""
    # Base config
    base_config: DefaultConfig
    
    # POMDP parameters (gнучка система)
    pomdp_params: Dict[str, Any]  # {"include_exposed": True, "noise": 0.1, "delay": 5, ...}
    
    # Scenario info
    scenario_name: str              # "mdp", "no_exposed", "custom"
    is_custom: bool
    
    # Agents to run
    target_agents: List[str]        # ["random", "threshold", "ppo_baseline", "ppo_framestack", "ppo_recurrent"]
    
    # Runtime settings
    train_rl: bool
    num_eval_episodes: int
    total_timesteps: int

class ExperimentDirectory:
    """Управління директоріями експерименту"""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.config = exp_config
        self.root = self._create_experiment_dir()
        
        # Subdirectories
        self.weights_dir = self.root / "weights"
        self.plots_dir = self.root / "plots"
        self.logs_dir = self.root / "logs"
        self.tensorboard_dir = self.logs_dir / "tensorboard"
        
    def _create_experiment_dir(self) -> Path:
        """Створює experiments/{scenario}/{timestamp}/ структуру"""
        
    def save_config(self) -> None:
        """Зберігає config.json з усіма параметрами"""
        
    def save_summary(self, results: List[SimulationResult]) -> None:
        """Зберігає summary.json з ключовими метриками"""
        
    def get_weight_path(self, agent_name: str) -> Path:
        """Повертає шлях до ваг: weights/{agent_name}.zip"""
        
    def get_plot_path(self, plot_name: str) -> Path:
        """Повертає шлях до графіка: plots/{plot_name}.png"""
        
    def get_log_path(self, agent_name: str) -> Path:
        """Повертає шлях до логу: logs/{agent_name}.txt"""

def load_experiment(experiment_path: str) -> ExperimentConfig:
    """Завантажує ExperimentConfig з існуючої папки експерименту"""
```

**Приклад config.json**:

```json
{
  "scenario_name": "no_exposed",
  "is_custom": false,
  "timestamp": "2026-02-05_14-30-00",
  "base_config": {
    "N": 100000,
    "E0": 200,
    "I0": 50,
    "beta_0": 0.4,
    "sigma": 0.2,
    "gamma": 0.1,
    "days": 200,
    "action_interval": 5,
    "w_I": 10,
    "w_S": 0.1,
    "thresholds": [0.01, 0.05, 0.09]
  },
  "pomdp_params": {
    "include_exposed": false
  },
  "target_agents": ["random", "threshold", "ppo_baseline"],
  "train_rl": true,
  "total_timesteps": 50000
}
```

### 2. `src/scenarios.py` (новий файл)

**Призначення**: Predefined сценарії для відтворюваності.

```python
PREDEFINED_SCENARIOS = {
    "mdp": {
        "description": "Baseline MDP (full observability, all target agents)",
        "pomdp_params": {
            "include_exposed": True,
        },
        "target_agents": ["random", "threshold", "ppo_baseline", "ppo_framestack", "ppo_recurrent"],
    },
    "no_exposed": {
        "description": "POMDP Experiment 1: Masked E compartment",
        "pomdp_params": {
            "include_exposed": False,
        },
        "target_agents": ["random", "threshold", "ppo_baseline", "ppo_framestack", "ppo_recurrent"],
    },
    # Майбутні експерименти:
    # "noisy_observations": {
    #     "description": "POMDP Experiment 2: Observations with Gaussian noise",
    #     "pomdp_params": {
    #         "include_exposed": True,
    #         "noise_std": 0.1,
    #     },
    #     "target_agents": [...],
    # },
}

def get_scenario(name: str) -> Dict[str, Any]:
    """Повертає конфігурацію predefined сценарію"""
    
def create_custom_scenario(pomdp_params: Dict[str, Any]) -> str:
    """Генерує назву для custom сценарію з параметрів"""
    # Приклад: {"include_exposed": False, "delay": 5} -> "no_exposed_delay5"
```

### 3. Оновлений `main.py`

**Зміни**:

- Замінити argparse на typer для кращого CLI
- Додати `--scenario` та динамічні прапорці (`--no-exposed`, `--delay`, `--noise` тощо)
- Додати `--load-experiment` для перезапуску
- Використовувати `ExperimentDirectory` для всіх операцій збереження

**Нова CLI структура**:

```bash
# Predefined сценарій
python main.py --scenario mdp --train-ppo

# Custom конфігурація
python main.py --no-exposed --train-ppo

# Перезапуск без навчання
python main.py --load-experiment experiments/mdp/2026-02-05_14-30-00/

# Майбутні параметри (легко додаються)
python main.py --no-exposed --delay 5 --noise 0.1 --train-ppo
```

**Ключові зміни в коді**:

- [`main.py:104`] Замінити `get_timestamped_results_dir()` на створення `ExperimentDirectory`
- [`main.py:26-32`] Оновити `train_and_plot_ppo()` для збереження в `exp_dir.weights_dir` та `exp_dir.plots_dir`
- [`main.py:35-57`] Оновити `load_ppo_agent()` для завантаження з `exp_dir.weights_dir`
- [`main.py:78-98`] Розширити argparse до typer з додатковими параметрами

### 4. Оновлений `src/train.py`

**Зміни**:

- Прийняти `ExperimentDirectory` як параметр
- Зберігати ваги в `exp_dir.get_weight_path(agent_name)`
- Зберігати TensorBoard логи в `exp_dir.tensorboard_dir`

**Сігнатура функції**:

```python
def train_ppo_agent(
    env_cls: Type[gym.Env],
    config: DefaultConfig,
    exp_dir: ExperimentDirectory,
    agent_name: str,  # "ppo_baseline", "ppo_framestack", "ppo_recurrent"
    total_timesteps: int,
) -> PPO:
```

### 5. Оновлений `src/evaluation.py`

**Зміни**:

- [`evaluation.py:95`] Замінити жорстко закодований `log_dir="logs"` на параметр
- [`evaluation.py:102`] Використовувати `exp_dir.get_plot_path()` та `exp_dir.get_log_path()`

### 6. Оновлений `src/utils.py`

**Зміни**:

- **Видалити** `get_timestamped_results_dir()` (переміщено в `ExperimentDirectory`)
- **Оновити** всі функції для використання consistent naming:
  - `plot_single_result()`: зберігає як `{agent_name}_seir.png`
  - `plot_all_results()`: зберігає як `comparison_all_agents.png`
  - `plot_learning_curve()`: зберігає як `{agent_name}_learning_episodes.png` та `{agent_name}_learning_timesteps.png`
  - `log_results()`: зберігає як `{agent_name}.txt`

### 7. `run_static_agents.py` (новий файл)

**Призначення**: Окремий скрипт для перевірки епідемічної моделі (не експеримент).

```bash
python run_static_agents.py
```

**Що робить**:

- Запускає 4 StaticAgent (NO, MILD, MODERATE, SEVERE)
- Створює один комбінований графік SEIR кривих
- Зберігає в `static_agents_results/{timestamp}/comparison.png` та конфігурацію.
- **Не зберігає** детальні логи, ваги, або індивідуальні графіки

## Consistency: Іменування файлів

### Графіки (всі в `plots/`)

- `comparison_all_agents.png` - загальний графік
- `{agent_name}_seir.png` - індивідуальні SEIR криві
  - Приклади: `random_agent_seir.png`, `threshold_agent_seir.png`, `ppo_baseline_seir.png`
- `{agent_name}_learning_episodes.png` - криві навчання по епізодах
- `{agent_name}_learning_timesteps.png` - криві навчання по timesteps

### Логи (всі в `logs/`)

- `{agent_name}.txt` - детальні логи дій
  - Приклади: `random_agent.txt`, `threshold_agent.txt`, `ppo_baseline.txt`

### Ваги (всі в `weights/`)

- `{agent_name}.zip` - ваги моделі
  - Приклади: `ppo_baseline.zip`, `ppo_framestack.zip`, `ppo_recurrent.zip`

## Гнучкість: Додавання нових параметрів

### Приклад додавання `--delay` параметру:

1. **Додати в `main.py` CLI**:

```python
@click.option("--delay", type=int, default=0, help="Observation delay in days")
```

1. **Додати в `pomdp_params**`:

```python
pomdp_params = {
    "include_exposed": args.include_exposed,
    "delay": args.delay,  # Автоматично потрапить в config.json
}
```

1. **Використовувати в wrapper** (коли імплементуватимете):

```python
if pomdp_params.get("delay", 0) > 0:
    env = DelayWrapper(env, delay=pomdp_params["delay"])
```

**Жодних змін** в `ExperimentDirectory`, `scenarios.py`, або системі збереження! Назва custom експерименту автоматично включить `delay5` якщо `delay != 0`.

## Перезапуск експериментів

```bash
# Завантажити і перезапустити (без навчання)
python main.py --load-experiment experiments/mdp/2026-02-05_14-30-00/
```

**Що відбувається**:

1. Завантажує `config.json`
2. Відтворює `ExperimentConfig`
3. Пропускає тренування (`train_rl=False`)
4. Завантажує ваги з `weights/`
5. Запускає evaluation
6. Зберігає в **новий** timestamped subdirectory того ж експерименту

## Міграція існуючих результатів

**Не потрібна**: старі `results/` та `logs/` папки можна залишити як є або видалити. Новий код створює чисту структуру з нуля.

## Порядок імплементації

1. Створити `src/experiment.py` з `ExperimentConfig` та `ExperimentDirectory`
2. Створити `src/scenarios.py` з `PREDEFINED_SCENARIOS`
3. Створити `run_static_agents.py` для окремої перевірки моделі
4. Оновити `src/utils.py` (видалити `get_timestamped_results_dir()`, consistent naming)
5. Оновити `src/train.py` для роботи з `ExperimentDirectory`
6. Оновити `src/evaluation.py` для роботи з `ExperimentDirectory`
7. Повністю переписати `main.py` з typer та новою логікою
8. Протестувати всі режими:
  - `--scenario mdp --train-ppo`
  - `--no-exposed --train-ppo`
  - `--load-experiment experiments/mdp/...`
  - `python run_static_agents.py`

## Важливі принципи

- **Простота**: Додавання нового параметру = 1 рядок в CLI + автоматичне збереження
- **Читабельність**: Чітка структура папок, self-documenting назви
- **Відтворюваність**: `config.json` містить ВСЕ для повного відтворення
- **Масштабованість**: Легко додати нові RL агенти, нові параметри, нові сценарії
- **Backwards compatibility**: Старий код не ламається, нова система працює паралельно

