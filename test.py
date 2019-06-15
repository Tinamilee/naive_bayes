from votesmart import votesmart
apikey = '49024thereoncewasamanfromnantucket94040'
bills = votesmart.votes.getBillsByStateRecent()
for bill in bills:
    print(bill.title, bill.billId)
