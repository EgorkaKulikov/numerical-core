# CLAUDE.md

Правила работы с этим репозиторием — в [AGENTS.md](AGENTS.md). Прочитайте его целиком
перед изменением файлов; этот файл не дублирует правила, чтобы не появилось второй
расходящейся копии.

Кратчайшая памятка:

- библиотека не импортирует `splines.*`, `solvers.*`, `problems.*`;
- после правки: `./gradlew test` (и `-Dnumerics.backend=reference` при правках
  линейной алгебры), перед отправкой: `./gradlew check`;
- правка формул → `publishToMavenLocal` → `characterizationTest` в `integral-equations`;
- новый алгоритм → строка в `docs/REFERENCES.md` с реальным источником.
