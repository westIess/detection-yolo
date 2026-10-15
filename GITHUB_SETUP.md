# Публикация проекта и следующие коммиты

## Подготовка

Распакуйте содержимое папки `ppe-detection-yolo` из ZIP в `C:\model Y` либо в новую рабочую папку. Если проект уже существует, предварительно сохраните его копию. Исходные скрипты в пакете сохранены; данные и промежуточные обучения не включены.

Откройте именно папку, в которой лежат `.gitignore` и `README.md`. Не загружайте ZIP как файл репозитория: GitHub должен получить распакованные файлы.

Проверьте в терминале `git --version`. Если Git не установлен, установите его с https://git-scm.com/downloads и перезапустите редактор.

## Через Visual Studio Code

1. **File → Open Folder** → папка проекта.
2. **Source Control** (`Ctrl+Shift+G`) → **Initialize Repository**.
3. Проверьте список изменений. В нём должны быть README, скрипты, графики и `runs/detect/train-16/weights/best.pt`. Датасета, окружения `.venv`, `last.pt` и прочих обучений быть не должно.
4. Добавьте нужные файлы в индекс кнопкой **+ / Stage All Changes**, введите сообщение `Initial commit: PPE detection baseline` и нажмите **Commit**.
5. Через `Ctrl+Shift+P` запустите **Publish to GitHub**, войдите в свой аккаунт, задайте имя `ppe-detection-yolo` и выберите видимость репозитория.
6. Откройте созданный репозиторий на GitHub: в README будут графики и пример детекции.

Для следующих изменений: сохраните файлы → просмотрите diff в Source Control → **Stage** → сообщение → **Commit** → **Push** (меню `…`). Commit сохраняет историю локально, Push отправляет её на GitHub.

Если Git попросит имя и почту, задайте их для этого проекта:

```powershell
git config user.name "Ваше имя"
git config user.email "Ваша почта или GitHub noreply-адрес"
```

## Через терминал VS Code или Visual Studio

Это альтернативный способ первой публикации. Не выполняйте его заново, если репозиторий уже опубликован через интерфейс.

Создайте на GitHub **пустой** репозиторий `ppe-detection-yolo`: не добавляйте README, лицензию и .gitignore, поскольку локально эти файлы уже подготовлены.

```powershell
cd "C:\model Y"
git init
git add .
git status
git diff --cached --stat
```

Проверьте состав коммита, затем:

```powershell
git commit -m "Initial commit: PPE detection baseline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ppe-detection-yolo.git
git push -u origin main
```

Замените `YOUR_USERNAME` своим логином. При запросе авторизации войдите в GitHub. Если `origin` уже существует, сначала посмотрите `git remote -v`; не добавляйте его повторно. Менять URL нужно только если текущий действительно неверен:

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/ppe-detection-yolo.git
```

## Следующие изменения

Во встроенном терминале любого из этих редакторов:

```powershell
git status
git diff
git add .
git diff --cached --stat
git commit -m "docs: update README"
git push
```

Сообщение коммита должно описывать конкретные изменения. При совместной работе сначала синхронизируйте ветку и разрешите возможные конфликты; не используйте force push для обхода ошибок.

## Почему картинка появится в README

Используется относительная ссылка:

```markdown
![Графики обучения](runs/detect/train-16/results.png)
```

Картинку и README нужно отправить в один репозиторий с сохранением структуры папок. Ссылка вида `https://runs/...` не работает.

## Проверка .gitignore

```powershell
git check-ignore -v datasets/example.jpg runs/detect/train-17/weights/best.pt yolo11n.pt
git ls-files runs/detect/train-16
```

После `git add .` вторая команда должна показать только выбранные веса, CSV, графики и конфигурацию. `.gitignore` открывает каждый родительский каталог `runs/detect/train-16`, поэтому нужные исключения работают.

Если хотите сохранить новый успешный запуск, добавьте аналогичные правила для его имени и обновите ссылки в README. По умолчанию новые запуски игнорируются.

## Если файлы уже были добавлены в Git

`.gitignore` не исключает ранее отслеживаемые файлы автоматически. Если данные и результаты лишь добавлены в индекс, но ещё не попали в коммит:

```powershell
git rm -r --cached --ignore-unmatch datasets runs
git add .
git status
```

`--cached` сохраняет локальные файлы. Выбранные результаты `train-16` добавятся обратно благодаря исключениям.

Если большой файл уже находится в прежнем коммите, новый коммит с удалением не удалит его из истории. Не повторяйте push вслепую и не удаляйте `.git`: потребуется исправить содержащий файл коммит либо аккуратно очистить историю с учётом остальных участников.

## Источники

- https://code.visualstudio.com/docs/sourcecontrol/repos-remotes
- https://git-scm.com/docs/gitignore
