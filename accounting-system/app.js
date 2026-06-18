// 从localStorage加载数据
let transactions = JSON.parse(localStorage.getItem('transactions')) || [];

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    // 设置默认日期为今天
    document.getElementById('date').value = new Date().toISOString().split('T')[0];
    
    // 渲染记录
    renderTransactions();
    updateSummary();
});

// 表单提交
document.getElementById('transactionForm').addEventListener('submit', (e) => {
    e.preventDefault();
    
    const type = document.getElementById('type').value;
    const description = document.getElementById('description').value;
    const amount = parseFloat(document.getElementById('amount').value);
    const date = document.getElementById('date').value;
    
    const transaction = {
        id: Date.now(),
        type,
        description,
        amount,
        date
    };
    
    transactions.unshift(transaction); // 添加到开头
    saveTransactions();
    renderTransactions();
    updateSummary();
    
    // 重置表单
    document.getElementById('transactionForm').reset();
    document.getElementById('date').value = new Date().toISOString().split('T')[0];
});

// 渲染记录列表
function renderTransactions() {
    const tbody = document.getElementById('recordsTable');
    const noRecords = document.getElementById('noRecords');
    
    if (transactions.length === 0) {
        tbody.innerHTML = '';
        noRecords.style.display = 'block';
        return;
    }
    
    noRecords.style.display = 'none';
    
    tbody.innerHTML = transactions.map(t => `
        <tr>
            <td>${t.date}</td>
            <td class="type-${t.type}">${t.type === 'income' ? '收入' : '支出'}</td>
            <td>${t.description}</td>
            <td class="amount-${t.type}">${t.type === 'income' ? '+' : '-'}¥${t.amount.toFixed(2)}</td>
            <td><button class="btn-delete" onclick="deleteTransaction(${t.id})">删除</button></td>
        </tr>
    `).join('');
}

// 删除记录
function deleteTransaction(id) {
    if (confirm('确定要删除这条记录吗？')) {
        transactions = transactions.filter(t => t.id !== id);
        saveTransactions();
        renderTransactions();
        updateSummary();
    }
}

// 更新统计
function updateSummary() {
    const income = transactions
        .filter(t => t.type === 'income')
        .reduce((sum, t) => sum + t.amount, 0);
    
    const expense = transactions
        .filter(t => t.type === 'expense')
        .reduce((sum, t) => sum + t.amount, 0);
    
    const balance = income - expense;
    
    document.getElementById('totalIncome').textContent = `¥${income.toFixed(2)}`;
    document.getElementById('totalExpense').textContent = `¥${expense.toFixed(2)}`;
    document.getElementById('balance').textContent = `¥${balance.toFixed(2)}`;
}

// 保存数据到localStorage
function saveTransactions() {
    localStorage.setItem('transactions', JSON.stringify(transactions));
}