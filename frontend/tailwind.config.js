export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        extend: {
            colors: {
                ink: "#080b14",
                panel: "#0f1424",
                neonCyan: "#22d3ee",
                neonBlue: "#3b82f6",
                neonPurple: "#a855f7"
            },
            boxShadow: {
                glass: "0 24px 50px rgba(8, 11, 20, 0.55)",
                neon: "0 0 22px rgba(56, 189, 248, 0.45)"
            },
            backgroundImage: {
                "grid-radial": "radial-gradient(circle at 20% 20%, rgba(59,130,246,0.2), transparent 40%), radial-gradient(circle at 80% 10%, rgba(168,85,247,0.18), transparent 35%), radial-gradient(circle at 50% 80%, rgba(34,211,238,0.16), transparent 40%)"
            }
        }
    },
    plugins: []
};
