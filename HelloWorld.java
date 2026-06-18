public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        // 1. 基本 for 循环：打印 0 到 4
        System.out.println("基本 for 循环：");
        for (int i = 0; i < 5; i++) {
            System.out.println("i = " + i);
        }

        // 2. 增强 for-each 循环：遍历数组
        System.out.println("\n增强 for 循环：");
        String[] names = {"Alice", "Bob", "Charlie"};
        for (String name : names) {
            System.out.println("name = " + name);
        }

        // 3. 倒序 for 循环
        System.out.println("\n倒序 for 循环：");
        for (int i = 5; i > 0; i--) {
            System.out.println("i = " + i);
        }

        // 4. 嵌套 for 循环：打印乘法表
        System.out.println("\n嵌套 for 循环（乘法表）：");
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                System.out.print(i * j + "\t");
            }
            System.out.println();
        }
    }
}
