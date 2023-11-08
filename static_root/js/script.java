import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class ChatboxApp {
    public static void main(String[] args) {
        JFrame frame = new JFrame("Chatbox");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 400);

        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());

        JTextArea textarea = new JTextArea();
        JPanel chatboxForm = new JPanel();

        textarea.setRows(1);

        textarea.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                updateTextArea();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                updateTextArea();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                // Not needed for plain text components
            }

            private void updateTextArea() {
                String text = textarea.getText();
                int lineCount = text.split("\n").length;
                textarea.setRows(Math.min(6, lineCount));
                if (textarea.getRows() > 1) {
                    chatboxForm.setAlignmentY(Component.BOTTOM_ALIGNMENT);
                } else {
                    chatboxForm.setAlignmentY(Component.CENTER_ALIGNMENT);
                }
            }
        });

        JButton chatboxToggle = new JButton("Toggle Chatbox");
        JPanel chatboxMessage = new JPanel();
        chatboxMessage.setVisible(false);

        chatboxToggle.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                chatboxMessage.setVisible(!chatboxMessage.isVisible());
            }
        });

        JButton dropdownToggle = new JButton("Dropdown Toggle");
        JPanel dropdownMenu = new JPanel();
        dropdownMenu.setVisible(false);

        dropdownToggle.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dropdownMenu.setVisible(!dropdownMenu.isVisible());
            }
        });

        frame.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (!e.getComponent().equals(dropdownMenu) && !e.getComponent().equals(dropdownToggle)) {
                    dropdownMenu.setVisible(false);
                }
            }
        });

        JPanel chatboxMessageWrapper = new JPanel();
        JPanel chatboxNoMessage = new JPanel();

        chatboxForm.setLayout(new FlowLayout(FlowLayout.LEFT));
        chatboxForm.add(textarea);

        chatboxMessage.setLayout(new BorderLayout());
        chatboxMessage.add(chatboxMessageWrapper, BorderLayout.CENTER);
        chatboxMessage.add(chatboxNoMessage, BorderLayout.SOUTH);

        frame.add(panel);
        panel.add(chatboxForm, BorderLayout.NORTH);
        panel.add(chatboxToggle, BorderLayout.WEST);
        panel.add(dropdownToggle, BorderLayout.EAST);
        panel.add(dropdownMenu, BorderLayout.SOUTH);
        panel.add(chatboxMessage, BorderLayout.CENTER);

        frame.setVisible(true);
    }
}
