# Best Practices for Branching, Feature Development, and Merging

## 1. Creating a Feature Branch and Merging via Pull Request

When developing a new feature, follow these best practices to ensure a clean and collaborative workflow.

### **Step 1: Ensure Your Local `main` is Up-to-Date**
Before creating a new branch, sync your local `main` branch with the latest code from the remote repository.

```bash
git checkout main  # Switch to the main branch
git pull origin main  # Fetch and merge the latest changes
```

### **Step 2: Create a Feature Branch**
Now, create a new branch for your feature development.

```bash
git checkout -b feature-branch  # Create and switch to the new branch
```

### **Step 3: Work on Your Feature**
Make your changes and commit them regularly.

```bash
git add .  # Stage all changes
git commit -m "Add feature description"  # Commit with a meaningful message
```

### **Step 4: Push Your Branch to Remote**
If collaborating with a team, push your feature branch to the remote repository.

```bash
git push -u origin feature-branch  # Push and track the branch
```

### **Step 5: Create a Pull Request (PR)**
1. Go to the repository on GitHub/GitLab/Bitbucket.
2. Navigate to **Pull Requests**.
3. Click **New Pull Request**.
4. Compare `feature-branch` with `main` and submit the PR.
5. Add a description and request a review from your team.

### **Step 6: Review and Merge**
1. Team members review your changes and provide feedback.
2. Address any requested changes.
3. Once approved, merge your PR into `main`.

### **Step 7: Delete Your Feature Branch**
After merging, clean up your local and remote branches.

```bash
git branch -d feature-branch  # Delete locally
git push origin --delete feature-branch  # Delete remotely
```

---

## 2. Keeping Your Feature Branch Up-to-Date with `main`
To avoid merge conflicts and ensure your feature is built on the latest `main`, regularly update your branch.

### **Option 1: Merge `main` into Your Feature Branch (Safer)**
```bash
git checkout main
git pull origin main
git checkout feature-branch
git merge main  # Merge latest main into your branch
```
This method keeps all commits but may create extra merge commits.


---

## 3. Undoing a Merge (Reverting a Merge)
If a merge causes issues, you can revert it.

```bash
git checkout main
git log  # Find the commit hash of the merge

git revert -m 1 <merge-commit-hash>  # Revert the merge commit

git push origin main  # Push the reverted changes
```

For a hard reset (only if you haven't pushed the merge yet):
```bash
git reset --hard <commit-before-merge>
```
**⚠ Warning:** This removes uncommitted changes.

---
